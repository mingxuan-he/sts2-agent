#!/bin/sh
# Immutable outer loop (lives in the image at /opt, NOT in /pod).
# All it knows how to do: seed /pod on first boot, run whatever
# /pod/loop.sh currently is, apply crash backoff and a daily session cap.
# It never inspects agent behavior — that's the (agent-editable) harness's job.
#
# Exit-code convention from the harness (informational only, except crash):
#   0 = session finished deliberately   2 = stalled   3 = token cap
#   anything else = crash -> exponential backoff

set -u

POD=/pod
SEED=/opt/seed
MAX_PER_DAY="${MAX_SESSIONS_PER_DAY:-50}"
BACKOFF=5
BACKOFF_MAX=3600

# First boot: seed the pod home
if [ ! -f "$POD/loop.sh" ]; then
    echo "[supervisor] seeding $POD from $SEED"
    cp -r "$SEED/." "$POD/"
fi

count_file="$POD/.supervisor/sessions-$(date -u +%Y%m%d)"
mkdir -p "$POD/.supervisor"

while true; do
    # Daily cap (UTC). Counter files are in /pod so the agent can SEE its
    # budget; editing them only wastes its own future sessions.
    today="$POD/.supervisor/sessions-$(date -u +%Y%m%d)"
    n=$(cat "$today" 2>/dev/null || echo 0)
    if [ "$n" -ge "$MAX_PER_DAY" ]; then
        echo "[supervisor] daily cap reached ($n/$MAX_PER_DAY), sleeping 1h"
        sleep 3600
        continue
    fi
    echo $((n + 1)) > "$today"

    echo "[supervisor] session $((n + 1))/$MAX_PER_DAY starting"
    sh "$POD/loop.sh"
    code=$?
    echo "[supervisor] session exited with code $code"

    case "$code" in
        0|2|3)
            BACKOFF=5
            sleep 5
            ;;
        *)
            echo "[supervisor] crash, backing off ${BACKOFF}s"
            sleep "$BACKOFF"
            BACKOFF=$((BACKOFF * 2))
            [ "$BACKOFF" -gt "$BACKOFF_MAX" ] && BACKOFF=$BACKOFF_MAX
            ;;
    esac
done
