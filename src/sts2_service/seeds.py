"""Seed pool generation.

Pools are generated once (random) and persisted in the DB, so they are stable
across service restarts. Seeds are visible to the agent (seed determinism is
the hidden flag); the EVAL pool is held out — its seeds are only ever assigned
in eval mode, and there is no endpoint that lists pool membership.
"""

from __future__ import annotations

import secrets
import string

TRAIN_POOL_SIZE = 200
EVAL_POOL_SIZE = 40

_ALPHABET = string.ascii_uppercase + string.digits


def _random_seed() -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(10))


def generate_pools(train: int = TRAIN_POOL_SIZE, eval_: int = EVAL_POOL_SIZE) -> list[tuple[str, str]]:
    seeds: set[str] = set()
    while len(seeds) < train + eval_:
        seeds.add(_random_seed())
    ordered = sorted(seeds)
    # random assignment of which seeds land in eval, so lexical position leaks nothing
    import random

    rng = random.SystemRandom()
    eval_set = set(rng.sample(ordered, eval_))
    return [(s, "eval" if s in eval_set else "train") for s in ordered]
