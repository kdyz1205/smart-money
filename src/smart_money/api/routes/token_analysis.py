"""Live token analysis endpoint — fetch real on-chain trades and run the full pipeline.

Fetches live trade data from GeckoTerminal and DexScreener public APIs,
converts to our Transaction model, and runs:
  1. Feature extraction → wallet profiling
  2. Clustering → behavioral grouping
  3. Anomaly detection → smart money identification
  4. Fill-speed analysis → high-speed accumulation alerts
  5. Volume surge detection → stealth accumulation signals
  6. Accumulation / coordinated buy prediction
"""

from __future__ import annotations

import logging
import os
import uuid
from collections import defaultdict
from datetime import datetime, timezone

import aiohttp
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...analyzer.features import extract_features
from ...analyzer.anomaly import detect_anomalies, compute_smart_money_score
from ...analyzer.clustering import cluster_wallets
from ...predictor.timeseries import detect_accumulation, detect_coordinated_buying
from ...shared.constants import Chain
from ...shared.models import (
    FillSpeedAlert,
    Transaction,
    VolumeSurge,
    WalletFeatures,
    WalletProfile,
)
from ...validator.fill_speed import analyze_fill_speed, detect_fill_speed_alerts
from ...validator.volume_filter import detect_volume_surges

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/token", tags=["token-analysis"])

# ── Constants ────────────────────────────────────────────────────────

GECKO_TERMINAL_BASE = "https://api.geckoterminal.com/api/v2"
DEXSCREENER_BASE = "https://api.dexscreener.com/latest/dex"

# Known pool addresses for CopperInu on Solana
COPPERINU_TOKEN = "61Wj56QgGyyB966T7YsMzEAKRLcMvJpDbPzjkrCZc4Bi"
COPPERINU_POOL = "3iUT1oAAUSqKeHAjkumZJbptihy2yN7AwFij2CrsUZVC"

NETWORK_TO_CHAIN: dict[str, Chain] = {
    "solana": Chain.SOL,
    "ethereum": Chain.ETH,
    "eth": Chain.ETH,
    "bsc": Chain.BSC,
    "arbitrum": Chain.ARB,
    "base": Chain.BASE,
    "polygon": Chain.POLYGON,
}


# ── Response models ──────────────────────────────────────────────────


class TokenOverview(BaseModel):
    token_address: str
    token_symbol: str
    chain: str
    price_usd: float
    market_cap_usd: float | None = None
    volume_24h_usd: float | None = None
    liquidity_usd: float | None = None
    price_change_24h_pct: float | None = None
    buys_24h: int = 0
    sells_24h: int = 0


class WalletAnalysis(BaseModel):
    address: str
    cluster_id: int | None = None
    is_smart_money: bool = False
    smart_money_score: float = 0.0
    tx_count: int = 0
    total_volume_usd: float = 0.0
    avg_tx_value_usd: float = 0.0
    unique_tokens: int = 0
    labels: list[str] = Field(default_factory=list)


class TradeActivity(BaseModel):
    total_trades: int = 0
    unique_wallets: int = 0
    buy_count: int = 0
    sell_count: int = 0
    total_buy_volume_usd: float = 0.0
    total_sell_volume_usd: float = 0.0
    net_flow_usd: float = 0.0
    avg_trade_usd: float = 0.0


class AccumulationInfo(BaseModel):
    detected: bool = False
    buy_probability: float = 0.0
    momentum_score: float = 0.0
    predicted_volume_usd: float = 0.0
    participating_wallets: int = 0


class CoordinatedBuyInfo(BaseModel):
    detected: bool = False
    coordinated_wallets: list[str] = Field(default_factory=list)
    count: int = 0


class FullAnalysisResponse(BaseModel):
    """Complete smart-money analysis for a token."""
    overview: TokenOverview
    trade_activity: TradeActivity
    smart_money_wallets: list[WalletAnalysis]
    all_wallets: list[WalletAnalysis]
    fill_speed_alerts: list[dict] = Field(default_factory=list)
    volume_surge: dict | None = None
    accumulation: AccumulationInfo
    coordinated_buying: CoordinatedBuyInfo
    analysis_timestamp: datetime
    summary: str


# ── Data fetching ────────────────────────────────────────────────────


async def _fetch_dexscreener_overview(token_address: str) -> dict | None:
    """Fetch token overview from DexScreener."""
    url = f"{DEXSCREENER_BASE}/tokens/{token_address}"
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status != 200:
                    logger.warning("DexScreener returned %d", resp.status)
                    return None
                try:
                    data = await resp.json()
                except (aiohttp.ContentTypeError, ValueError):
                    logger.warning("DexScreener returned non-JSON response")
                    return None
                pairs = data.get("pairs") or []
                if not pairs:
                    return None
                # Return the pair with highest volume (guard against empty after filter)
                valid_pairs = [p for p in pairs if isinstance(p, dict)]
                if not valid_pairs:
                    return None
                return max(valid_pairs, key=lambda p: float(p.get("volume", {}).get("h24", 0) or 0))
    except Exception as e:
        logger.error("DexScreener fetch failed: %s", e)
        return None


async def _fetch_gecko_trades(network: str, pool_address: str) -> list[dict]:
    """Fetch recent trades from GeckoTerminal."""
    url = f"{GECKO_TERMINAL_BASE}/networks/{network}/pools/{pool_address}/trades"
    all_trades: list[dict] = []
    max_trades = 500  # cap to prevent memory explosion
    try:
        async with aiohttp.ClientSession(trust_env=True) as session:
            # Fetch multiple pages for more data
            for _page in range(3):
                params = {"trade_volume_in_usd_greater_than": "10"}
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    if resp.status != 200:
                        break
                    try:
                        data = await resp.json()
                    except (aiohttp.ContentTypeError, ValueError):
                        break
                    trades = data.get("data") or []
                    if not trades:
                        break
                    all_trades.extend(trades)
                    if len(all_trades) >= max_trades:
                        all_trades = all_trades[:max_trades]
                        break
                    # GeckoTerminal doesn't always paginate the same way
                    break
    except Exception as e:
        logger.error("GeckoTerminal trades fetch failed: %s", e)
    return all_trades


def _gecko_trades_to_transactions(
    trades: list[dict], token_address: str, token_symbol: str, network: str = "solana"
) -> list[Transaction]:
    """Convert GeckoTerminal trade data to our Transaction model."""
    chain = NETWORK_TO_CHAIN.get(network, Chain.SOL)
    txs: list[Transaction] = []
    for i, trade in enumerate(trades):
        attrs = trade.get("attributes", {})
        tx_hash = attrs.get("tx_hash", str(uuid.uuid4()))
        kind = attrs.get("kind", "buy")  # "buy" or "sell"
        volume_usd = float(attrs.get("volume_in_usd", "0") or "0")
        block = int(attrs.get("block_number", 0) or 0)
        from_addr = attrs.get("tx_from_address", "unknown")

        # Parse timestamp (GeckoTerminal uses ISO 8601 or Unix ms)
        ts_str = attrs.get("block_timestamp")
        ts = None
        if ts_str:
            try:
                if isinstance(ts_str, (int, float)):
                    # Auto-detect seconds vs milliseconds
                    val = float(ts_str)
                    if val > 1e12:
                        val = val / 1000
                    ts = datetime.fromtimestamp(val, tz=timezone.utc)
                elif isinstance(ts_str, str) and ts_str.isdigit():
                    val = int(ts_str)
                    if val > 1e12:
                        val = val // 1000
                    ts = datetime.fromtimestamp(val, tz=timezone.utc)
                elif isinstance(ts_str, str):
                    ts = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
            except (ValueError, TypeError, OSError):
                logger.debug("Failed to parse timestamp: %s", ts_str)
        if ts is None:
            ts = datetime.now(timezone.utc)

        # Convert volume to a wei-like integer (scale by 1e18 for pipeline compatibility)
        value_wei = int(volume_usd * 1e12)  # scale factor for analysis

        # For buys: from_addr is the buyer (sending SOL to pool)
        # For sells: from_addr is the seller (sending tokens to pool)
        tx_from = from_addr if from_addr and from_addr != "unknown" else f"anon_{i}"
        tx_to = token_address if kind == "buy" else f"pool_{token_address[:8]}"

        txs.append(
            Transaction(
                tx_hash=tx_hash,
                chain=chain,
                from_addr=tx_from.lower() if tx_from else "unknown",
                to_addr=tx_to.lower() if tx_to else "unknown",
                value_wei=value_wei,
                token_symbol=token_symbol,
                token_address=token_address,
                block_number=block,
                timestamp=ts,
                gas_used=5000,  # Solana compute units placeholder
                method_id="swap" if kind == "buy" else None,
            )
        )
    return txs


# ── Analysis pipeline ────────────────────────────────────────────────


def _run_wallet_analysis(
    txs: list[Transaction],
    token_address: str = "",
) -> tuple[list[WalletProfile], list[WalletFeatures], list[WalletAnalysis]]:
    """Extract features, cluster, detect anomalies, score wallets."""
    features = extract_features(txs)
    if not features:
        return [], [], []

    # Infer chain from transactions
    tx_chain = txs[0].chain if txs else Chain.SOL

    # Cluster
    profiles = cluster_wallets(features, min_samples=min(3, len(features)), chain=tx_chain)

    # Anomaly detection
    anomalies = detect_anomalies(features, contamination=0.1)
    anomaly_map = {feat.address.lower(): score for feat, score in anomalies}
    feat_map = {f.address.lower(): f for f in features}

    for profile in profiles:
        key = profile.address.lower()
        a_score = anomaly_map.get(key, 0.0)
        feat = feat_map.get(key)
        if feat:
            profile.smart_money_score = compute_smart_money_score(
                anomaly_score=a_score,
                win_rate=feat.win_rate,
                volume_usd=feat.total_volume_usd,
            )
            profile.is_smart_money = profile.smart_money_score >= 0.3  # lower threshold for meme coins
            if profile.is_smart_money:
                profile.labels.append("smart_money")

    # Filter out contract/pool addresses and the token contract itself
    exclude_addrs = {
        addr.lower() for addr in [
            "unknown", "pool", token_address,
        ] if addr
    }
    # Also exclude any address that looks like a pool or contract placeholder
    for f in features:
        if f.address.startswith("pool_") or f.address.startswith("anon_"):
            exclude_addrs.add(f.address)

    # Build wallet analysis list
    wallet_analyses = []
    for profile in profiles:
        if profile.address.lower() in exclude_addrs:
            continue
        feat = feat_map.get(profile.address.lower())
        wallet_analyses.append(
            WalletAnalysis(
                address=profile.address,
                cluster_id=profile.cluster_id,
                is_smart_money=profile.is_smart_money,
                smart_money_score=round(profile.smart_money_score, 4),
                tx_count=int(feat.tx_frequency_7d) if feat else 0,
                total_volume_usd=round(feat.total_volume_usd / 1e12, 2) if feat else 0,
                avg_tx_value_usd=round(feat.avg_tx_value_usd / 1e12, 2) if feat else 0,
                unique_tokens=feat.unique_tokens_traded if feat else 0,
                labels=profile.labels,
            )
        )

    # Sort: smart money first, then by score
    wallet_analyses.sort(key=lambda w: (-int(w.is_smart_money), -w.smart_money_score))

    return profiles, features, wallet_analyses


def _run_fill_speed_analysis(
    txs: list[Transaction], profiles: list[WalletProfile], token_address: str, liquidity_usd: float
) -> list[dict]:
    """Run fill-speed alert detection for smart money wallets."""
    smart_addrs = {p.address for p in profiles if p.is_smart_money}
    alerts_out: list[dict] = []

    for addr in smart_addrs:
        wallet_txs = [tx for tx in txs if tx.from_addr.lower() == addr]
        if len(wallet_txs) < 2:
            continue

        by_token: dict[str, list[Transaction]] = defaultdict(list)
        for tx in wallet_txs:
            key = tx.token_address or tx.to_addr
            by_token[key].append(tx)

        alerts = detect_fill_speed_alerts(
            wallet_txs_by_token=by_token,
            wallet_address=addr,
            historical_speeds=[],
            market_volumes={token_address: liquidity_usd * 1e12},
            token_liquidity={token_address: liquidity_usd * 1e12},
            rapid_interval_sec=120.0,  # wider window for Solana
        )

        for alert in alerts:
            alerts_out.append({
                "wallet": alert.wallet_address,
                "fill_speed_usd_per_sec": round(alert.fill_speed_usd_per_sec / 1e12, 4),
                "stealth_score": round(alert.stealth_score, 2),
                "num_rapid_trades": alert.num_rapid_trades,
                "avg_interval_sec": round(alert.avg_interval_sec, 1),
                "liquidity_pct": round(alert.liquidity_pct, 4),
                "trigger": alert.metadata.get("trigger", "unknown"),
            })

    return alerts_out


def _run_volume_surge_analysis(
    txs: list[Transaction],
    profiles: list[WalletProfile],
    volume_24h_usd: float,
    price_usd: float,
    price_change_pct: float,
    liquidity_usd: float,
) -> dict | None:
    """Check for volume surge from smart-money wallets."""
    smart_addrs = {p.address for p in profiles if p.is_smart_money}
    if not smart_addrs or not txs:
        return None

    # Approximate price at window start
    price_at_start = price_usd / (1 + price_change_pct / 100) if price_change_pct else price_usd

    surge = detect_volume_surges(
        txs=txs,
        smart_money_addresses=smart_addrs,
        total_market_volume_usd=liquidity_usd * 1e12,
        avg_volume_24h_usd=volume_24h_usd * 1e12,
        current_price=price_usd,
        price_at_window_start=price_at_start,
        window_minutes=15,
        sm_ratio_threshold=0.20,  # lower for meme coins
    )

    if not surge:
        return None

    return {
        "detected": True,
        "sm_volume_ratio": surge.sm_volume_ratio,
        "vs_24h_avg_multiplier": surge.vs_24h_avg_multiplier,
        "net_buy_volume_usd": round(surge.net_buy_volume_usd / 1e12, 2),
        "price_change_pct": surge.price_change_pct,
        "is_stealth_accumulation": surge.is_stealth_accumulation,
        "contributing_wallets": surge.contributing_wallets[:10],
    }


def _run_accumulation_detection(
    txs: list[Transaction], profiles: list[WalletProfile], token_address: str, token_symbol: str
) -> tuple[AccumulationInfo, CoordinatedBuyInfo]:
    """Run accumulation and coordinated buying detection."""
    smart_addrs = {p.address for p in profiles if p.is_smart_money}

    buy_vols = [float(tx.value_wei) for tx in txs if tx.from_addr in smart_addrs]
    sell_vols = [float(tx.value_wei) for tx in txs if tx.from_addr not in smart_addrs]
    timestamps = [tx.timestamp.timestamp() for tx in txs]
    wallet_addrs = list({tx.from_addr for tx in txs if tx.from_addr in smart_addrs})

    acc_info = AccumulationInfo()
    acc = detect_accumulation(
        token_address=token_address,
        token_symbol=token_symbol,
        buy_volumes=buy_vols,
        sell_volumes=sell_vols,
        timestamps=timestamps,
        wallet_addresses=wallet_addrs,
        window=min(12, max(3, len(buy_vols))),
    )
    if acc:
        acc_info = AccumulationInfo(
            detected=True,
            buy_probability=round(acc.buy_probability, 4),
            momentum_score=round(acc.momentum_score, 4),
            predicted_volume_usd=round(acc.predicted_volume_usd / 1e12, 2),
            participating_wallets=len(acc.wallet_addresses),
        )

    # Coordinated buying
    coord_info = CoordinatedBuyInfo()
    wallet_buy_times: dict[str, list[float]] = defaultdict(list)
    for tx in txs:
        if tx.from_addr in smart_addrs:
            wallet_buy_times[tx.from_addr].append(tx.timestamp.timestamp())

    coordinated = detect_coordinated_buying(
        wallet_buy_times, time_window_sec=600.0, min_wallets=2
    )
    if coordinated:
        coord_info = CoordinatedBuyInfo(
            detected=True,
            coordinated_wallets=coordinated[:10],
            count=len(coordinated),
        )

    return acc_info, coord_info


# ── Main endpoint ────────────────────────────────────────────────────


@router.get("/analyze/copperinu", response_model=FullAnalysisResponse)
async def analyze_copperinu() -> FullAnalysisResponse:
    """Shortcut: analyze CopperInu token (Solana)."""
    return await analyze_token(
        token_address=COPPERINU_TOKEN,
        network="solana",
        pool=COPPERINU_POOL,
    )


@router.get("/analyze/{token_address}", response_model=FullAnalysisResponse)
async def analyze_token(
    token_address: str,
    network: str = Query(default="solana", description="Blockchain network"),
    pool: str = Query(default="", description="Pool address (auto-detected if empty)"),
) -> FullAnalysisResponse:
    """Run full smart-money analysis on a live token.

    Fetches real-time trades from DexScreener + GeckoTerminal,
    then runs the complete analysis pipeline.
    """
    # Step 1: Fetch token overview from DexScreener
    overview_data = await _fetch_dexscreener_overview(token_address)
    if not overview_data:
        raise HTTPException(status_code=404, detail=f"Token {token_address} not found on DexScreener")

    pair_addr = overview_data.get("pairAddress", pool)
    base_token = overview_data.get("baseToken", {})
    token_symbol = base_token.get("symbol", "UNKNOWN")
    price_usd = float(overview_data.get("priceUsd", 0) or 0)
    volume_24h = float(overview_data.get("volume", {}).get("h24", 0) or 0)
    liquidity_usd = float(overview_data.get("liquidity", {}).get("usd", 0) or 0)
    market_cap = float(overview_data.get("marketCap", 0) or overview_data.get("fdv", 0) or 0)
    price_change_24h = float(overview_data.get("priceChange", {}).get("h24", 0) or 0)
    txns = overview_data.get("txns", {}).get("h24", {})
    buys_24h = int(txns.get("buys", 0) or 0)
    sells_24h = int(txns.get("sells", 0) or 0)

    chain_id = overview_data.get("chainId", network)

    overview = TokenOverview(
        token_address=token_address,
        token_symbol=token_symbol,
        chain=chain_id,
        price_usd=price_usd,
        market_cap_usd=market_cap,
        volume_24h_usd=volume_24h,
        liquidity_usd=liquidity_usd,
        price_change_24h_pct=price_change_24h,
        buys_24h=buys_24h,
        sells_24h=sells_24h,
    )

    # Step 2: Fetch trades from GeckoTerminal
    raw_trades = await _fetch_gecko_trades(network, pair_addr)
    if not raw_trades:
        raise HTTPException(
            status_code=502,
            detail="Could not fetch trade data from GeckoTerminal. Try again later.",
        )

    # Step 3: Convert to Transaction model
    txs = _gecko_trades_to_transactions(raw_trades, token_address, token_symbol, network)
    logger.info("Fetched %d trades for %s (%s)", len(txs), token_symbol, token_address[:16])

    # Trade activity summary
    buy_txs = [tx for tx in txs if tx.method_id == "swap"]
    sell_txs = [tx for tx in txs if tx.method_id is None]
    total_buy_vol = sum(tx.value_wei for tx in buy_txs) / 1e12
    total_sell_vol = sum(tx.value_wei for tx in sell_txs) / 1e12

    trade_activity = TradeActivity(
        total_trades=len(txs),
        unique_wallets=len({tx.from_addr for tx in txs}),
        buy_count=len(buy_txs),
        sell_count=len(sell_txs),
        total_buy_volume_usd=round(total_buy_vol, 2),
        total_sell_volume_usd=round(total_sell_vol, 2),
        net_flow_usd=round(total_buy_vol - total_sell_vol, 2),
        avg_trade_usd=round((total_buy_vol + total_sell_vol) / max(len(txs), 1), 2),
    )

    # Step 4: Run wallet analysis pipeline
    profiles, features, wallet_analyses = _run_wallet_analysis(txs, token_address=token_address)
    smart_wallets = [w for w in wallet_analyses if w.is_smart_money]

    # Step 5: Fill speed alerts
    fill_alerts = _run_fill_speed_analysis(txs, profiles, token_address, liquidity_usd)

    # Step 6: Volume surge
    vol_surge = _run_volume_surge_analysis(
        txs, profiles, volume_24h, price_usd, price_change_24h, liquidity_usd
    )

    # Step 7: Accumulation & coordinated buying
    acc_info, coord_info = _run_accumulation_detection(txs, profiles, token_address, token_symbol)

    # Step 8: Generate summary
    summary_parts = [
        f"Analyzed {len(txs)} recent trades across {trade_activity.unique_wallets} wallets for {token_symbol}.",
        f"Price: ${price_usd:.6f} | MCap: ${market_cap:,.0f} | 24h Vol: ${volume_24h:,.0f} | Liq: ${liquidity_usd:,.0f}",
        f"Identified {len(smart_wallets)}/{len(wallet_analyses)} wallets as potential smart money.",
    ]
    if fill_alerts:
        summary_parts.append(f"⚠ {len(fill_alerts)} fill-speed alert(s) detected — possible high-speed accumulation.")
    if vol_surge:
        summary_parts.append(
            f"⚠ Volume surge detected: SM ratio {vol_surge['sm_volume_ratio']:.1%}"
            + (" (STEALTH)" if vol_surge["is_stealth_accumulation"] else "")
        )
    if acc_info.detected:
        summary_parts.append(
            f"📈 Accumulation pattern: {acc_info.buy_probability:.0%} buy probability, "
            f"momentum {acc_info.momentum_score:+.3f}"
        )
    if coord_info.detected:
        summary_parts.append(
            f"🤝 Coordinated buying: {coord_info.count} wallets acting together"
        )
    if not fill_alerts and not vol_surge and not acc_info.detected:
        summary_parts.append("No strong smart-money signals detected in the current window.")

    return FullAnalysisResponse(
        overview=overview,
        trade_activity=trade_activity,
        smart_money_wallets=smart_wallets,
        all_wallets=wallet_analyses,
        fill_speed_alerts=fill_alerts,
        volume_surge=vol_surge,
        accumulation=acc_info,
        coordinated_buying=coord_info,
        analysis_timestamp=datetime.now(timezone.utc),
        summary="\n".join(summary_parts),
    )