"""Headless STS2 environment wrapper.

Spawns sts2-cli as a subprocess, communicates via JSON stdin/stdout.
Supports parallel instances for RL rollouts — each instance is an
independent game process with no shared state.
"""

import asyncio
import json
import os
import subprocess
from pathlib import Path
from typing import Any

STS2_CLI_DIR = Path(__file__).parent / "sts2-cli"
DOTNET = os.environ.get("DOTNET_PATH", os.path.expanduser("~/.dotnet/dotnet"))
PROJECT = str(STS2_CLI_DIR / "src" / "Sts2Headless" / "Sts2Headless.csproj")


class HeadlessEnv:
    """Single headless STS2 game instance."""

    def __init__(self, character: str = "Ironclad", ascension: int = 0,
                 seed: str | None = None, lang: str = "en"):
        self.character = character
        self.ascension = ascension
        self.seed = seed
        self.lang = lang
        self._proc: asyncio.subprocess.Process | None = None
        self._lock = asyncio.Lock()

    async def start(self) -> dict[str, Any]:
        """Launch the C# process and start a run. Returns initial decision point."""
        self._proc = await asyncio.create_subprocess_exec(
            DOTNET, "run", "--project", PROJECT, "-c", "Release",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(STS2_CLI_DIR),
            env={**os.environ, "DOTNET_CLI_TELEMETRY_OPTOUT": "1"},
        )
        # Read the ready message
        ready = await self._read()
        assert ready.get("type") == "ready", f"Expected ready, got {ready}"

        # Start a run
        cmd = {
            "cmd": "start_run",
            "character": self.character,
            "ascension": self.ascension,
            "lang": self.lang,
        }
        if self.seed is not None:
            cmd["seed"] = self.seed
        return await self._send(cmd)

    async def action(self, action: str, args: dict[str, Any] | None = None) -> dict[str, Any]:
        """Send a game action (play_card, end_turn, select_map_node, etc.)."""
        cmd: dict[str, Any] = {"cmd": "action", "action": action}
        if args:
            cmd["args"] = args
        return await self._send(cmd)

    async def get_map(self) -> dict[str, Any]:
        """Get the full map layout."""
        return await self._send({"cmd": "get_map"})

    async def quit(self) -> dict[str, Any]:
        """Quit the current run and terminate the process."""
        result = await self._send({"cmd": "quit"})
        await self.close()
        return result

    async def close(self):
        """Kill the subprocess."""
        if self._proc and self._proc.returncode is None:
            self._proc.kill()
            await self._proc.wait()
        self._proc = None

    async def _send(self, cmd: dict) -> dict[str, Any]:
        async with self._lock:
            line = json.dumps(cmd) + "\n"
            self._proc.stdin.write(line.encode())
            await self._proc.stdin.drain()
            return await self._read()

    async def _read(self) -> dict[str, Any]:
        # `dotnet run` can emit MSBuild/restore chatter on stdout before the
        # protocol starts; every real protocol line is a JSON object, so skip
        # anything that isn't one.
        for _ in range(200):
            line = await self._proc.stdout.readline()
            if not line:
                stderr = await self._proc.stderr.read()
                raise RuntimeError(f"sts2-cli process died: {stderr.decode()[-500:]}")
            stripped = line.strip()
            if stripped.startswith(b"{"):
                return json.loads(stripped)
        raise RuntimeError("sts2-cli produced no JSON output in 200 lines")

    @property
    def is_alive(self) -> bool:
        return self._proc is not None and self._proc.returncode is None


class HeadlessEnvPool:
    """Pool of parallel headless environments for RL rollouts."""

    def __init__(self, n: int, character: str = "Ironclad", ascension: int = 0):
        self.envs = [HeadlessEnv(character=character, ascension=ascension) for _ in range(n)]

    async def start_all(self) -> list[dict[str, Any]]:
        return await asyncio.gather(*(env.start() for env in self.envs))

    async def close_all(self):
        await asyncio.gather(*(env.close() for env in self.envs))


# ── Quick test ──
async def _smoke_test():
    env = HeadlessEnv(character="Ironclad", seed="smoketest", ascension=0)
    print("Starting headless STS2...")
    state = await env.start()
    print(f"Decision: {state['decision']}")
    print(f"Character: {state['player']['name']}, HP: {state['player']['hp']}/{state['player']['max_hp']}")

    # Pick first Neow option
    if state["decision"] == "event_choice":
        print(f"Neow options: {[o['title'] for o in state['options']]}")
        state = await env.action("choose_option", {"option_index": 0})
        print(f"After Neow: {state['decision']}")

    # Navigate to first map node
    if state["decision"] == "map_select":
        choices = state["choices"]
        print(f"Map choices: {[(c['col'], c['row'], c['type']) for c in choices]}")
        state = await env.action("select_map_node", {"col": choices[0]["col"], "row": choices[0]["row"]})
        print(f"Entered: {state['decision']}")

    # If combat, play one turn
    if state["decision"] == "combat_play":
        print(f"Combat! Energy: {state['energy']}, Hand: {[c['name'] for c in state['hand']]}")
        print(f"Enemies: {[(e['name'], e['hp']) for e in state['enemies']]}")

        # Play all playable cards
        for card in state["hand"]:
            if card["can_play"]:
                args = {"card_index": card["index"]}
                if card["target_type"] == "AnyEnemy":
                    args["target_index"] = 0
                state = await env.action("play_card", args)
                decision = state.get("decision")
                if decision != "combat_play":
                    print(f"  Played {card['name']} → {decision or state.get('type')}")
                    break
                print(f"  Played {card['name']}, enemies: {[(e['name'], e['hp']) for e in state.get('enemies', [])]}")

        if state.get("decision") == "combat_play":
            state = await env.action("end_turn")
            print(f"After end turn: {state.get('decision')}")

    result = await env.quit()
    print(f"Done: {result}")


if __name__ == "__main__":
    asyncio.run(_smoke_test())
