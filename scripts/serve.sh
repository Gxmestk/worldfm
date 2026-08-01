#!/usr/bin/env bash
# Self-guarding WorldFM live-loop webserver (aiohttp WebSocket, step=1, reuse-only).
# Boots live_server.py and kills its OWN process group if RAM_avail < RAM_LIMIT
# (protects the 16GB cgroup, no swap; kill by PID/group — NEVER pkill -f).
#
#   Run:  bash scripts/serve.sh              (foreground; Ctrl-C to stop)
#   View: http://<container-or-host>:8123/   (WS /stream · GET /frame · /metrics)
#   Env:  PORT=8123 HOST=0.0.0.0 STEP=1 RAM_LIMIT=2500 bash scripts/serve.sh
set -m          # job control -> backgrounded job gets its own process group (clean pgroup kill)
set -uo pipefail
cd /root/gtk-projects/world-model/worldfm || exit 1

# REQUIRED (hard rule #5): the NVIDIA runtime libs live inside the uv venv, not a
# system path; without this cuDNN/nvrtc JIT fails ("No execution plans support the graph").
source .venv/bin/activate
NVIDIA_LIBS=$(find "$PWD/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"

PORT=${PORT:-8123}; HOST=${HOST:-0.0.0.0}; STEP=${STEP:-1}
RAM_LIMIT=${RAM_LIMIT:-2500}
mkdir -p logs
LOG=logs/serve.log; : > "$LOG"
VRAM=logs/serve_vram.log; : > "$VRAM"

cleanup () { [ -n "${PY:-}" ] && kill -9 -"$PY" 2>/dev/null; }   # negative PID = whole pgroup
trap cleanup INT TERM

{
  echo "=== $(date) LIVE SERVER START host=$HOST port=$PORT step=$STEP ram_limit=${RAM_LIMIT}MiB ==="
  uv run python live_server.py --meta demo/meta.json --host "$HOST" --port "$PORT" --step "$STEP" &
  PY=$!
  while kill -0 "$PY" 2>/dev/null; do
    ram=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null)
    nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$VRAM"
    if [ -n "$ram" ] && [ "$ram" -lt "$RAM_LIMIT" ] 2>/dev/null; then
      echo "$(date) SELF-ABORT: RAM avail=${ram}MiB < $RAM_LIMIT -> kill -9 -$PY (pgroup)"
      kill -9 -"$PY" 2>/dev/null; sleep 2; break
    fi
    sleep 1
  done
  wait "$PY" 2>/dev/null
  echo "=== $(date) LIVE SERVER STOPPED ==="
  awk '$1 ~ /^[0-9]+$/{m=($1>m?$1:m); n++} END{print "peak_VRAM_MiB", m+0, "samples", n+0}' "$VRAM"
} >> "$LOG" 2>&1
