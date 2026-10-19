#!/bin/bash
# Job script for generate_cpu.sub: run_cpu.sh <repo> <n> <out> <job> <seed0> <batch>
set -euo pipefail
repo=$1 n=$2 out=$3 job=$4 seed0=$5 batch=$6
mkdir -p "$(dirname "$out")"
echo "$(hostname)  $(nproc) cpus visible, OMP_NUM_THREADS=${OMP_NUM_THREADS:-unset}  $(date)"
t0=$(date +%s)
python "$repo/generate.py" --n "$n" --out "$out" --seed $((seed0 + job)) --batch "$batch"
echo "finished $n showers in $(( $(date +%s) - t0 )) s"
