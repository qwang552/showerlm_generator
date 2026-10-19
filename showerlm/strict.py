"""Light-front and end-time masks used while sampling."""
from __future__ import annotations

import numpy as np
import torch

from .common import C_CM_PER_NS, VOX_CM
from .tokens import KIND_CONT, KIND_STROKE

FRONT_LAG_MIN = -0.0175
T_END_MAX = 12.5
COUNTS = {}


class Strict:
    def __init__(self, vocab, lag_min=FRONT_LAG_MIN, t_end_max=T_END_MAX):
        self.v = vocab
        self.lag_min = None if lag_min is None else float(lag_min)
        self.t_end_max = None if t_end_max is None else float(t_end_max)
        self.ns_per_vox = VOX_CM / C_CM_PER_NS
        self.off = torch.as_tensor(vocab.dec_off(np.arange(vocab.n_off)), dtype=torch.float64)
        self.len_ns = torch.as_tensor(vocab.dec_len(np.arange(vocab.n_len)) * self.ns_per_vox, dtype=torch.float64)

    def tc(self, tb):
        return (tb.double() + 0.5) * self.v.t_max / self.v.n_t

    def apply(self, name, lg, tok, ai):
        if name == "oz" and self.lag_min is not None:
            iz = tok[:, ai["iz"]].double()
            need = self.v.g - iz - 0.5 - (self.tc(tok[:, ai["t"]]) - self.lag_min) / self.ns_per_vox
            ok = self.off.to(lg.device)[None, : lg.shape[1]] >= need[:, None] - 1e-9
            top = lg.shape[1] - 1
        elif name == "len" and self.t_end_max is not None:
            room = self.t_end_max - self.tc(tok[:, ai["t"]])
            ok = self.len_ns.to(lg.device)[None, : lg.shape[1]] <= room[:, None] + 1e-9
            ok[:, 0] = True
            top = 0
        else:
            return lg
        k = tok[:, ai["kind"]]
        ok = ok | ~((k == KIND_STROKE) | (k == KIND_CONT))[:, None]
        fin = torch.isfinite(lg)
        cut = (fin & ~ok).any(1)
        COUNTS[name] = COUNTS.get(name, 0) + int(cut.sum())
        none = ~(ok & fin).any(1)
        if none.any():
            COUNTS[name + "_fallback"] = COUNTS.get(name + "_fallback", 0) + int(none.sum())
            ok = ok.clone()
            ok[none, top] = True
            lg = lg.clone()
            lg[none, top] = torch.where(torch.isfinite(lg[none, top]), lg[none, top], torch.zeros_like(lg[none, top]))
        return lg.masked_fill(~ok, float("-inf"))
