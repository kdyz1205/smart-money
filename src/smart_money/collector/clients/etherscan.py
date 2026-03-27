"""Etherscan API client for fetching Ethereum on-chain data."""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from ...shared.constants import Chain
from ...shared.models import TokenTransfer, Transaction
from .base import BaseBlockchainClient

logger = logging.getLogger(__name__)

_ETHERSCAN_API = "https://api.etherscan.io/api"


class EtherscanClient(BaseBlockchainClient):
    """Fetches transactions and token transfers from Etherscan."""

    def __init__(self, api_key: str, chain: Chain = Chain.ETH) -> None:
        super().__init__(base_url=_ETHERSCAN_API, api_key=api_key)
        self._chain = chain

    async def get_transactions(
        self, address: str, start_block: int = 0
    ) -> list[Transaction]:
        data = await self._request(
            {
                "module": "account",
                "action": "txlist",
                "address": address,
                "startblock": start_block,
                "endblock": 99999999,
                "sort": "desc",
                "apikey": self._api_key,
            }
        )
        results: list[Transaction] = []
        for item in data.get("result", []):
            if not isinstance(item, dict):
                continue
            results.append(
                Transaction(
                    tx_hash=item["hash"],
                    chain=self._chain,
                    from_addr=item["from"],
                    to_addr=item.get("to", ""),
                    value_wei=int(item["value"]),
                    block_number=int(item["blockNumber"]),
                    timestamp=datetime.fromtimestamp(
                        int(item["timeStamp"]), tz=timezone.utc
                    ),
                    gas_used=int(item.get("gasUsed", 0)),
                    method_id=item.get("methodId"),
                )
            )
        logger.info("Fetched %d transactions for %s", len(results), address)
        return results

    async def get_token_transfers(
        self, address: str, start_block: int = 0
    ) -> list[TokenTransfer]:
        data = await self._request(
            {
                "module": "account",
                "action": "tokentx",
                "address": address,
                "startblock": start_block,
                "endblock": 99999999,
                "sort": "desc",
                "apikey": self._api_key,
            }
        )
        results: list[TokenTransfer] = []
        for item in data.get("result", []):
            if not isinstance(item, dict):
                continue
            decimals = int(item.get("tokenDecimal", 18))
            raw_value = int(item.get("value", 0))
            results.append(
                TokenTransfer(
                    tx_hash=item["hash"],
                    chain=self._chain,
                    from_addr=item["from"],
                    to_addr=item.get("to", ""),
                    token_address=item["contractAddress"],
                    token_symbol=item.get("tokenSymbol", "UNKNOWN"),
                    amount=raw_value / (10**decimals),
                    timestamp=datetime.fromtimestamp(
                        int(item["timeStamp"]), tz=timezone.utc
                    ),
                )
            )
        logger.info("Fetched %d token transfers for %s", len(results), address)
        return results
