#!/bin/bash
# Sequential only -- never two JAX sims at once (an unsliced parallel run once OOM'd the machine).
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}   # point PYTHON at a JAX-enabled interpreter
export SLAM_OBS_SLICE=800
for course in circuit circuit_alias; do
  for arm in full noplace noplace_nogrid noplace_nogrid_noring; do
    echo "=== $arm / $course ==="
    if [ "$course" = "circuit_alias" ]; then export SLAM_ALIAS_KFOLD=2; else unset SLAM_ALIAS_KFOLD; fi
    $PY minimal_stack/minimal_sufficient_stack.py --arm "$arm" --course "$course" --seeds 12 --steps 600 \
      2>&1 | grep -E "seed|wrote|Error|Traceback"
  done
done
echo "ALL ARMS COMPLETE"
