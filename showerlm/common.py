"""Shared constants and small helpers."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

C_CM_PER_NS = 29.9792458
C_M_PER_NS = 0.299792458
VOX_CM = 400.0 / 192.0
YIELD_CM = 343.0
N_ICE = 1.31


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def read_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def fresh_dir(base, name):
    base = Path(base)
    out = base / name
    k = 2
    while out.exists():
        out = base / f"{name}_{k}"
        k += 1
    out.mkdir(parents=True)
    return out
