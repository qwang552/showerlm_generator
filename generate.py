#!/usr/bin/env python
"""Generate CORSIKA 8-like in-ice Cherenkov light showers with ShowerLM (no training data needed).

    python generate.py --n 10 --out showers --seed 0

Writes showers/sample_XXXXX.npz (voxels: ijk, q, off, m, t, cnt) and tokens_XXXXX.npz, plus run.json, timing.csv,
summary.json.  Uses the GPU when one is available, otherwise the CPU.
"""
import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from showerlm.sample04 import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=10, help="number of showers")
    ap.add_argument("--out", default="showers", help="output folder (must not exist)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=10, help="showers generated in parallel")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--no-strict", action="store_true", help="switch off the strict light-front / end-time sampling rules")
    ap.add_argument("--no-t-caps", action="store_true", help="switch off the caps on the detail time steps")
    ap.add_argument("--models", default=str(HERE / "trained_model"), help="folder with skeleton/ and detail/")
    a = ap.parse_args()
    out = Path(a.out).resolve()
    if out.exists():
        raise SystemExit(f"{out} exists (not overwritten)")
    m = Path(a.models)
    run(m / "skeleton", m / "detail", a.n, min(a.batch, a.n), out.parent, out.name, seed=a.seed, temperature=a.temperature,
        allow_cpu=True, strict=not a.no_strict, t_caps=not a.no_t_caps)


if __name__ == "__main__":
    main()
