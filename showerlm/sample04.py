"""Generation from noise: skeleton model, then detail model, then rendering."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from .common import fresh_dir, log, write_json
from .raster import render
from .sample import generate, load_run
from .strict import FRONT_LAG_MIN, T_END_MAX, Strict
from .train import device_for, is_detail, seq_len_at


T_JUMP_MAX = 212
T_FIRST_MAX = 165


def merge(tok_skel, tok_det, vocab):
    ix = vocab.index
    d = np.asarray(tok_det, dtype=np.int64).reshape(-1, len(vocab.names)).copy()
    for a in ("ptr", "s", "dx", "dy", "dz"):
        d[:, ix[a]] = 0
    return np.concatenate([np.asarray(tok_skel, dtype=np.int64).reshape(-1, len(vocab.names)), d])


def save04(out, i, tok_skel, tok_det, vocab, ranges, seconds):
    tok = merge(tok_skel, tok_det, vocab)
    r = render(vocab.decode(tok, use_pointer=False), vocab.g, ranges)
    np.savez_compressed(out / f"sample_{i:05d}.npz", ijk=r["ijk"], q=r["q"], off=r["off"], m=r["m"], t=r["t"], cnt=r["cnt"],
                        seconds=np.asarray([seconds], np.float32))
    np.savez_compressed(out / f"tokens_{i:05d}.npz", tok=tok.astype(np.int16), tok_skel=np.asarray(tok_skel).astype(np.int16),
                        tok_det=np.asarray(tok_det).astype(np.int16))
    return r


def run(skel_run, det_run, n, batch, out, name, seed=0, temperature=1.0, allow_cpu=True, strict=True, t_caps=True,
        det_plan_run=None, skel_max_tokens=1024, max_tokens=20000, rebuild_every=256, ranges=(-2, 2, -2, 2, 46, 50)):
    from .detail import generate_detail
    device = device_for(allow_cpu)
    det_model, vocab, det_cfg, det_ck = load_run(det_run, device, "best_stroke")
    plan_model, plan_ck = None, det_ck
    plan_dir = det_plan_run or det_run
    if (Path(plan_dir) / "best_plan.pt").exists():
        plan_model, plan_vocab, _, plan_ck = load_run(plan_dir, device, "best_plan")
        if plan_vocab.to_dict() != vocab.to_dict():
            raise SystemExit("the plan checkpoint and the detail model use different vocabularies")
    if not is_detail(det_cfg):
        raise SystemExit(f"{det_run} is not a detail model")
    k_max = int(det_cfg["model"].get("k_max", 1024))
    det_chunk = min(seq_len_at(det_cfg["train"]["seq_len_schedule"], 10 ** 9), int(det_cfg["model"]["window"]))
    skel_model, skel_vocab, skel_cfg, skel_ck = load_run(skel_run, device, "best")
    if skel_vocab.to_dict() != vocab.to_dict():
        raise SystemExit("the skeleton and detail models use different vocabularies")
    if strict:
        for m in (skel_model, det_model):
            m.strict = Strict(vocab, FRONT_LAG_MIN, T_END_MAX)
    out = fresh_dir(out, name)
    ranges = np.asarray(ranges, dtype=np.float64).reshape(3, 2)
    log(f"skeleton {skel_ck}; detail strokes {det_ck}; detail plan {plan_ck}; on {device}; {n} showers -> {out}")
    write_json(out / "run.json", dict(skel_checkpoint=str(skel_ck), det_checkpoint=str(det_ck), det_plan_checkpoint=str(plan_ck), n=n, batch=batch,
                                      temperature=temperature, seed=seed, strict=bool(strict), t_jump_max=T_JUMP_MAX if t_caps else None,
                                      t_first_max=T_FIRST_MAX if t_caps else None, device=str(device), vocab=vocab.to_dict(), ranges=ranges.tolist()))
    rows = []
    i = 0
    while i < n:
        B = min(batch, n - i)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        skels, sk_plan = generate(skel_model, vocab, B, device, temperature, min(skel_max_tokens, k_max), 10 ** 9, seed=seed * 100003 + i,
                                  verbose=False, birth_ptr=bool(skel_cfg["data"].get("birth_ptr", True)))
        t1 = time.time()
        dets, plan = generate_detail(det_model, vocab, skels, device, temperature, max_tokens, det_chunk, rebuild_every, seed=seed * 100003 + i + 7,
                                     verbose=True, plan_model=plan_model, t_jump_max=T_JUMP_MAX if t_caps else None,
                                     t_first_max=T_FIRST_MAX if t_caps else None)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.time()
        sec = (t2 - t0) / B
        for b in range(B):
            r = save04(out, i + b, skels[b], dets[b], vocab, ranges, sec)
            rows.append(dict(index=i + b, n_skel=int(len(skels[b])), n_det=int(len(dets[b])), det_stop=plan["stop"][b],
                             n_vox=int(len(r["q"])), q_total=float(r["q"].sum()), seconds=round(sec, 2)))
        log(f"  {i + B}/{n}: {sec:.1f} s/shower; last: {rows[-1]}")
        i += B
    cols = list(rows[0])
    with open(out / "timing.csv", "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join(str(r[k]) for k in cols) + "\n")
    write_json(out / "summary.json", dict(n=len(rows), n_vox_mean=float(np.mean([r["n_vox"] for r in rows])),
                                          q_total_mean=float(np.mean([r["q_total"] for r in rows])),
                                          seconds_per_shower=float(np.mean([r["seconds"] for r in rows]))))
    log("done")
    return out, rows
