"""OKX DEX API client for fetching DEX trading data."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ...shared.constants import Chain
from ...shared.models import TokenTransfer, Transaction
from .base import BaseBlockchainClient

logger = logging.getLogger(__name__)

_OKX_DEX_API = "https://www.okx.com/api/v5/dex/aggregator"

# OKX chain ID mapping
_CHAIN_IDS: dict[Chain, str] = {
    Chain.ETH: "1",
    Chain.BSC: "56",
    Chain.ARB: "42161",
    Chain.BASE: "8453",
    Chain.POLYGON: "137",
    Chain.SOL: "501",
}


class OkxDexClient(BaseBlockchainClient):
    """Fetches DEX trading data from OKX aggregator API."""

    def __init__(self, api_key: str, chain: Chain = Chain.ETH) -> None:
        super().__init__(base_url=_OKX_DEX_API, api_key=api_key)
        self._chain = chain
        self._chain_id = _CHAIN_IDS.get(chain, "1")

    async def get_transactions(
        self, address: str, start_block: int = 0
    ) -> list[Transaction]:
        """Fetch swap transactions for a wallet from OKX DEX aggregator."""
        data = await self._request(
            {
                "chainId": self._chain_id,
                "userAddr": address,
            }
        )
        results: list[Transaction] = []
        for item in data.get("data", []):
            if not isinstance(item, dict):
                continue
            results.append(
                Transaction(
                    tx_hash=item.get("txHash", ""),
                    chain=self._chain,
                    from_addr=address,
                    to_addr=item.get("toTokenAddress", ""),
                    value_wei=int(float(item.get("fromTokenAmount", 0))),
                    token_symbol=item.get("toTokenSymbol"),
                    token_address=item.get("toTokenAddress"),
                    block_number=int(item.get("blockNumber", 0)),
                    timestamp=datetime.fromtimestamp(
                        int(item.get("timestamp", 0)) / 1000, tz=timezone.utc
                    ),
                    gas_used=int(float(item.get("gasUsed", 0))),
                    method_id=item.get("methodId"),
                )
            )
        logger.info("OKX DEX: fetched %d txs for %s", len(results), address)
        return results

    async def get_token_transfers(
        self, address: str, start_block: int = 0
    ) -> list[TokenTransfer]:
        """OKX DEX does not provide raw token transfer events — returns empty."""
        return []
