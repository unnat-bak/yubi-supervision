#!/usr/bin/env bash
# Integration smoke test: health → start → poll status → stream → stop.
# Requires a running server (./scripts/dev.sh) and a webcam-permitted process.
set -euo pipefail

BASE="${BASE:-http://127.0.0.1:8000}"
TIMEOUT="${TIMEOUT:-90}"

fail() { echo "FAIL: $1" >&2; exit 1; }

echo "1/5 health"
health=$(curl -fsS "$BASE/api/health") || fail "health endpoint unreachable"
echo "$health" | grep -q '"status":"ok"' || fail "health not ok: $health"

echo "2/5 start"
start=$(curl -fsS -X POST "$BASE/api/start") || fail "start failed"
echo "$start" | grep -Eq '"state":"(starting|live)"' || fail "unexpected start state: $start"

echo "3/5 poll status until live (timeout ${TIMEOUT}s)"
deadline=$((SECONDS + TIMEOUT))
state="starting"
while [[ $SECONDS -lt $deadline ]]; do
  status=$(curl -fsS "$BASE/api/status")
  state=$(echo "$status" | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")
  if [[ "$state" == "live" ]]; then
    echo "  live: $(echo "$status" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"fps={d['fps']} latency_ms={d['latency_ms']} degraded={d['degraded']}\")")"
    break
  fi
  if [[ "$state" == "error" ]]; then
    fail "engine errored: $(echo "$status" | python3 -c "import sys,json; print(json.load(sys.stdin)['error'])")"
  fi
  sleep 1
done
[[ "$state" == "live" ]] || fail "did not reach live within ${TIMEOUT}s (state=$state)"

echo "4/5 stream returns frames"
bytes=$( (curl -fsS --max-time 5 "$BASE/api/stream" 2>/dev/null || true) | head -c 20000 | wc -c | tr -d ' ')
[[ "$bytes" -gt 1000 ]] || fail "stream returned only $bytes bytes"
echo "  got $bytes bytes of MJPEG"

echo "5/5 stop"
stop=$(curl -fsS -X POST "$BASE/api/stop") || fail "stop failed"
echo "$stop" | grep -q '"state":"idle"' || fail "unexpected stop state: $stop"
final=$(curl -fsS "$BASE/api/status")
echo "$final" | grep -q '"state":"idle"' || fail "status not idle after stop: $final"

echo "PASS: full lifecycle ok"
