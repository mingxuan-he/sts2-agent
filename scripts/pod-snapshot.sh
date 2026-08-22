#!/bin/bash
# Snapshot the pod: /pod volume (the agent's entire accumulated context) plus a
# consistent copy of the game DB. Restore + eval flow comes in phase 3.
#
# Usage: scripts/pod-snapshot.sh [snapshot-id]     (from repo root)
set -euo pipefail

ID="${1:-$(date -u +%Y%m%dT%H%M%S)}"
OUT="snapshots/$ID"
mkdir -p "$OUT"

echo "snapshotting pod volume -> $OUT/pod.tgz"
docker run --rm -v pod_pod_home:/pod:ro -v "$PWD/$OUT":/out alpine \
    tar czf /out/pod.tgz -C /pod .

echo "backing up game DB -> $OUT/sts2.sqlite3"
docker exec pod-game-1 /venv/bin/python -c "
import sqlite3
src = sqlite3.connect('/data/sts2.sqlite3')
dst = sqlite3.connect('/data/snapshot-tmp.sqlite3')
src.backup(dst); dst.close(); src.close()"
docker cp pod-game-1:/data/snapshot-tmp.sqlite3 "$OUT/sts2.sqlite3"
docker exec pod-game-1 rm /data/snapshot-tmp.sqlite3

echo "done: $OUT"
ls -lh "$OUT"
