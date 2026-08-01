#!/usr/bin/env bash
# Self-guarding WorldFM inference: step=2 then step=1, reusing cached anchors.
# Kills its OWN python PID if RAM_avail < RAM_LIMIT (protects the 16GB cgroup).
# Preserves each step's performance.json + PNG frames before the next run clobbers them.
set -uo pipefail
cd /root/gtk-projects/world-model/worldfm || exit 1
source .venv/bin/activate
NVIDIA_LIBS=$(find "$PWD/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"
mkdir -p logs; LOG=logs/runs.log; : > "$LOG"
RAM_LIMIT=${RAM_LIMIT:-2500}

run_step () {
  local step="$1"
  local vram=logs/vram_run${step}.log; : > "$vram"
  {
    echo "=== $(date) RUN START step=$step (reuse_intermediates, profile_worldfm, save_mode=image) ==="
    python run_pipeline.py --meta demo/meta.json --output_dir outputs --step "$step" \
        --reuse_intermediates --profile_worldfm --save_mode image &
    local PY=$!
    while kill -0 "$PY" 2>/dev/null; do
      ram=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null)
      nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$vram"
      if [ -n "$ram" ] && [ "$ram" -lt "$RAM_LIMIT" ] 2>/dev/null; then
        echo "$(date) SELF-ABORT step=$step: RAM avail=${ram}MiB < $RAM_LIMIT -> kill -9 $PY"
        kill -9 "$PY" 2>/dev/null; sleep 2; break
      fi
      sleep 1
    done
    wait "$PY" 2>/dev/null
    echo "=== $(date) RUN END step=$step ==="
    if [ -f outputs/mario/performance.json ]; then
      cp outputs/mario/performance.json outputs/mario/performance_step${step}.json
      echo "saved outputs/mario/performance_step${step}.json"
    fi
    for f in outputs/mario/output_*.png; do [ -e "$f" ] && mv "$f" "outputs/mario/step${step}_$(basename "$f")"; done
    python -c "v=[int(l) for l in open('$vram') if l.strip()]; print('peak_VRAM_MiB', max(v) if v else 0,'samples',len(v))"
  } >> "$LOG" 2>&1
}

run_step 2
run_step 1
{ echo "=== $(date) ALL RUNS DONE ==="; echo "--- outputs/mario ---"; ls -la outputs/mario/; } >> "$LOG" 2>&1
