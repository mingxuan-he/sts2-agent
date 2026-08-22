"""Async client for the STS2MCP mod HTTP API."""

from __future__ import annotations

import httpx
from dataclasses import dataclass, field
from typing import Any


@dataclass
class STS2Client:
    """Thin async wrapper around the STS2MCP mod's REST API.

    The mod runs an HTTP server on the game host. In production this is
    Ming's Mac exposed via Tailscale; for RL training we'll need a headless
    game instance (or a simulator — see README).
    """

    base_url: str = "http://localhost:15526"
    mode: str = "singleplayer"  # "singleplayer" or "multiplayer"
    timeout: float = 30.0
    _client: httpx.AsyncClient = field(init=False, repr=False)

    def __post_init__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
        )

    @property
    def _endpoint(self) -> str:
        return f"/api/v1/{self.mode}"

    async def get_state(self, fmt: str = "json") -> dict[str, Any]:
        """Get the current game state."""
        resp = await self._client.get(self._endpoint, params={"format": fmt})
        resp.raise_for_status()
        return resp.json()

    async def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        """Send an action and return the resulting game state."""
        resp = await self._client.post(
            self._endpoint,
            json=action,
        )
        resp.raise_for_status()
        return resp.json()

    async def play_card(self, card_index: int, target: str | None = None) -> dict[str, Any]:
        action = {"action": "play_card", "card_index": card_index}
        if target is not None:
            action["target"] = target
        return await self.send_action(action)

    async def end_turn(self) -> dict[str, Any]:
        return await self.send_action({"action": "end_turn"})

    async def proceed(self) -> dict[str, Any]:
        return await self.send_action({"action": "proceed"})

    async def choose_map_node(self, index: int) -> dict[str, Any]:
        return await self.send_action({"action": "choose_map_node", "index": index})

    async def select_card_reward(self, card_index: int) -> dict[str, Any]:
        return await self.send_action({"action": "select_card_reward", "card_index": card_index})

    async def skip_card_reward(self) -> dict[str, Any]:
        return await self.send_action({"action": "skip_card_reward"})

    async def claim_reward(self, index: int) -> dict[str, Any]:
        return await self.send_action({"action": "claim_reward", "index": index})

    async def choose_rest_option(self, index: int) -> dict[str, Any]:
        return await self.send_action({"action": "choose_rest_option", "index": index})

    async def choose_event_option(self, index: int) -> dict[str, Any]:
        return await self.send_action({"action": "choose_event_option", "index": index})

    async def use_potion(self, slot: int, target: str | None = None) -> dict[str, Any]:
        action = {"action": "use_potion", "slot": slot}
        if target is not None:
            action["target"] = target
        return await self.send_action(action)

    async def shop_purchase(self, index: int) -> dict[str, Any]:
        return await self.send_action({"action": "shop_purchase", "index": index})

    async def close(self):
        await self._client.aclose()
