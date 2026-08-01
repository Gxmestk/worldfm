#!/usr/bin/env bash
# Self-guarding prepare: runs run_pipeline --prepare_only and kills its OWN python PID
# if system RAM_avail drops below RAM_LIMIT (protects this 16GB VM). Kills by PID, never
# by pattern, so it cannot self-kill. Persistent logs in logs/ (in-repo, not /tmp).
cd /root/gtk-projects/world-model/worldfm || exit 1
source .venv/bin/activate
NVIDIA_LIBS=$(find "$PWD/.venv/lib/python3.10/site-packages/nvidia" -name lib -type d 2>/dev/null | tr '\n' ':')
export LD_LIBRARY_PATH="${NVIDIA_LIBS}${LD_LIBRARY_PATH}"
mkdir -p logs
LOG=logs/prepare.log; : > "$LOG"
VRAMLOG=logs/vram_prepare.log; : > "$VRAMLOG"
RAM_LIMIT=${RAM_LIMIT:-2500}  # MiB avail floor; self-abort below this (kernel cgroup OOM is at ~0)

echo "=== $(date) PREPARE START (nf4, 512x1024, pipe.to(cuda), self-guard RAM_LIMIT=${RAM_LIMIT}MiB) ===" | tee -a "$LOG"
python run_pipeline.py --meta demo/meta.json --output_dir outputs --prepare_only >> "$LOG" 2>&1 &
PY=$!
while kill -0 "$PY" 2>/dev/null; do
  ram=$(awk '/MemAvailable/{print int($2/1024)}' /proc/meminfo 2>/dev/null)
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null >> "$VRAMLOG"
  if [ -n "$ram" ] && [ "$ram" -lt "$RAM_LIMIT" ] 2>/dev/null; then
    { echo "$(date) SELF-ABORT: RAM avail=${ram}MiB < ${RAM_LIMIT} -> kill -9 $PY"; } | tee -a "$LOG"
    kill -9 "$PY" 2>/dev/null
    sleep 2
    break
  fi
  sleep 1
done
wait "$PY" 2>/dev/null
echo "=== $(date) PREPARE END ===" | tee -a "$LOG"
echo "--- intermediates ---"; ls -la outputs/mario/intermediates/ 2>&1 | tee -a "$LOG"
python -c "v=[int(l) for l in open('$VRAMLOG') if l.strip()]; print('peak_VRAM_MiB', max(v) if v else 0,'samples',len(v))" | tee -a "$LOG"
