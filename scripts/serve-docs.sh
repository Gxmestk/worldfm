#!/usr/bin/env bash
# Serve the mkdocs-material docs site (mkdocs.yml at repo root, source in docs/).
#
#   Run:  bash scripts/serve-docs.sh         (foreground; Ctrl-C to stop)
#   View: http://<container-or-host>:8126/   (auto-rebuilds on docs/*.md edits)
#   Env:  PORT=8126 HOST=0.0.0.0 bash scripts/serve-docs.sh
#   Log:  tail -f logs/docs.log
#
# NOTE: we call the project venv's interpreter directly (`.venv/bin/python -m mkdocs`).
# Do NOT switch to `uv run mkdocs` — in this container `uv run` resolves to
# /opt/venv/0/bin/python3, which has no mkdocs; only the project's uv-managed `.venv`
# (Python 3.10) does (installed via requirements-docs.txt). mkdocs is pure-Python, so
# unlike scripts/serve.sh it needs no CUDA LD_LIBRARY_PATH and no RAM self-guard.
set -m          # job control -> backgrounded mkdocs + its livereload watcher share one pgroup
set -uo pipefail
cd /root/gtk-projects/world-model/worldfm || exit 1

PORT=${PORT:-8126}; HOST=${HOST:-0.0.0.0}
mkdir -p logs
LOG=logs/docs.log; : > "$LOG"

cleanup () { [ -n "${PY:-}" ] && kill -9 -"$PY" 2>/dev/null; }   # negative PID = whole pgroup
trap cleanup INT TERM

{
  echo "=== $(date) MKDOCS SERVE host=$HOST port=$PORT ==="
  .venv/bin/python -m mkdocs serve --dev-addr "$HOST:$PORT" &
  PY=$!
  wait "$PY" 2>/dev/null
  echo "=== $(date) MKDOCS SERVE STOPPED ==="
} >> "$LOG" 2>&1
