"""Sampling rules applied as masks on the attribute logits."""
from __future__ import annotations

import math

import numpy as np
import torch

from .common import C_CM_PER_NS, VOX_CM
from .tokens import KIND_CONT, KIND_EOS, KIND_PAD, KIND_STROKE

RULES = ("t_cap", "time_order", "cone", "cont_window", "lt", "kind", "q_per_record", "support", "hung_cone")
N_REF = 1.31
Q_PER_ROW = 343.5
LT_LEN_MIN = 58
CONT_SLACK_VOX = 0.25 + 1.891 + math.sqrt(3.0) / 8.0
D_MAX = 8
SAMPLE_FALLBACKS = {}


class Rules:

    def __init__(self, vocab, cfg, model, detail=False):
        cfg = dict(cfg or {})
        bad = set(cfg) - set(RULES)
        if bad:
            raise ValueError(f"unknown rules {sorted(bad)}; known: {RULES}")
        self.on = {k: bool(cfg.get(k, False)) for k in RULES}
        self.any = any(self.on.values())
        self.v, self.detail = vocab, bool(detail)
        self.ai = vocab.index
        self.t_cap = int(vocab.enc_t(12.0 - 1e-9))
        self.t_win_lo = int(getattr(model, "t_win_lo", 0))
        self.cont = bool(getattr(model, "cont", False))
        dt = vocab.t_max / vocab.n_t
        self.dt = dt

        self.X = torch.as_tensor(C_CM_PER_NS * (np.arange(vocab.n_t) + 1.0) * dt / VOX_CM, dtype=torch.float64)

        L = vocab.dec_len(np.arange(vocab.n_len)) + CONT_SLACK_VOX
        self.cont_top = torch.as_tensor(np.floor(N_REF * L * VOX_CM / C_CM_PER_NS / dt) + 1 + int(getattr(model, "cont_top_extra", 0)), dtype=torch.long)

        self.t_rule = torch.as_tensor(vocab.t_bins_of_len(np.arange(vocab.n_len)), dtype=torch.long) if vocab.v3 else None

        lq = np.arange(vocab.n_q); lc = np.arange(vocab.n_c)
        q_lo = 10.0 ** (lq * vocab.q_log_max / vocab.n_q)
        r_hi = 10.0 ** ((lc + 1) * vocab.c_log_max / vocab.n_c)
        ok = (q_lo[:, None] <= Q_PER_ROW * r_hi[None, :] * (1 + 1e-9)) | (lc[None, :] == vocab.n_c - 1)
        self.lc_ok = torch.as_tensor(ok)
        db = np.arange(vocab.sizes[self.ai["dx"]])
        self.d_ok = torch.as_tensor((db >= 1) & (np.abs(vocab.dec_d(db)) <= D_MAX))
        self.s_ok = torch.as_tensor(np.arange(vocab.sizes[self.ai["s"]]) >= 1)
        self._dev = None

    def to(self, device):
        if self._dev != device:
            for k in ("X", "cont_top", "t_rule", "lc_ok", "d_ok", "s_ok"):
                if getattr(self, k) is not None:
                    setattr(self, k, getattr(self, k).to(device))
            self._dev = device
        return self


    def kind(self, n_kind, prev_lt, has_prev, t_rule=None, prev_tok=None):
        N = has_prev.shape[0]
        b = torch.arange(n_kind, device=has_prev.device)[None, :].expand(N, n_kind)
        ok = b != KIND_PAD
        if not self.on["kind"]:
            return ok
        if self.detail or n_kind <= KIND_CONT:
            return ok & (b != KIND_CONT)
        force = has_prev & (prev_lt > 0)
        ok = torch.where(force[:, None], b == KIND_CONT, ok)
        no_cont = ~has_prev
        if t_rule is not None and self.on["cont_window"] and self.on["t_cap"]:
            no_cont |= ~force & (t_rule - self.t_win_lo > self.t_cap)
        if prev_tok is not None and self.on["hung_cone"] and self.on["cont_window"]:
            no_cont |= ~force & has_prev & ~self.cont_reachable(prev_tok)
        ok &= ~no_cont[:, None] | (b != KIND_CONT)
        return ok

    def t_floor(self, is_cont, t_order, t_rule):
        f = torch.zeros_like(is_cont, dtype=torch.long)
        if self.on["time_order"] and t_order is not None:
            f = torch.where(is_cont, f, t_order)
        if self.on["cont_window"] and t_rule is not None:
            f = torch.where(is_cont, (t_rule - self.t_win_lo).clamp_min(0), f)
        return f.clamp(0, self.v.n_t - 1)

    def t(self, n_t, is_cont, t_order, t_rule, prev_t, prev_len):
        N = is_cont.shape[0]
        b = torch.arange(n_t, device=is_cont.device)[None, :]
        ok = torch.ones(N, n_t, dtype=torch.bool, device=is_cont.device)
        if self.on["t_cap"]:
            ok &= b <= self.t_cap
        ok &= b >= self.t_floor(is_cont, t_order, t_rule)[:, None]
        if self.on["cont_window"] and self.cont and prev_t is not None:
            top = prev_t + self.cont_top[prev_len.clamp(0, len(self.cont_top) - 1)]
            ok &= ~is_cont[:, None] | (b <= top[:, None])
        return ok

    def _g(self, i, lo_edge):
        if lo_edge == "xy":
            h = self.v.g // 2
            return torch.clamp(torch.maximum(i - h, (h - 1) - i), min=0).double()
        return (self.v.g - 1 - i).clamp_min(0).double()

    def position(self, name, n, t, ix=None, iy=None):
        N = t.shape[0]
        dev = t.device
        if not self.on["cone"]:
            return torch.ones(N, n, dtype=torch.bool, device=dev)
        b = torch.arange(n, device=dev)[None, :].expand(N, n)
        X2 = self.X[t.clamp(0, self.v.n_t - 1)][:, None] ** 2 + 1e-6
        if name == "ix":
            return self._g(b, "xy") ** 2 <= X2
        gx2 = self._g(ix, "xy")[:, None] ** 2
        if name == "iy":
            return gx2 + self._g(b, "xy") ** 2 <= X2
        gy2 = self._g(iy, "xy")[:, None] ** 2
        return gx2 + gy2 + self._g(b, "z") ** 2 <= X2


    def anchors(self, parent, s_bin):
        v, ai = self.v, self.ai
        par = parent.detach().cpu().numpy().astype(np.int64)
        head = v.head_point(par)
        u = v.dec_dir(par[:, ai["th"]], par[:, ai["ph"]])
        L = v.dec_len(par[:, ai["len"]])
        sv = np.where(L > 0, v.dec_s(s_bin.detach().cpu().numpy()), 0.0)
        a = np.clip(np.floor(head + (sv * L)[:, None] * u).astype(np.int64), 0, v.g - 1)
        return torch.from_numpy(a).to(parent.device)

    def _dset(self, device):
        nb = self.v.sizes[self.ai["dx"]]
        ok = self.d_ok.cpu().numpy() if self.on["support"] else np.ones(nb, bool)
        return torch.as_tensor(np.unique(self.v.dec_d(np.arange(nb)[ok])), device=device)

    def _axis_min(self, anc, axis):
        c = (anc[:, axis:axis + 1] + self._dset(anc.device)[None, :]).clamp(0, self.v.g - 1)
        return self._g(c, "z" if axis == 2 else "xy").min(1, keepdim=True).values

    def box_g2(self, anc):
        return (self._axis_min(anc, 0) ** 2 + self._axis_min(anc, 1) ** 2 + self._axis_min(anc, 2) ** 2)[:, 0]

    def piece_g2(self, pieces):
        M = pieces.shape[0]
        S = self.v.sizes[self.ai["s"]]
        if M == 0:
            return torch.zeros(0, S, dtype=torch.float64, device=pieces.device)
        par = pieces[:, None, :].expand(M, S, pieces.shape[1]).reshape(M * S, -1)
        sb = torch.arange(S, device=pieces.device).repeat(M)
        out = self.box_g2(self.anchors(par, sb)).reshape(M, S)
        if self.on["support"]:
            out[:, 0] = float("inf")
        return out

    def cont_reachable(self, piece):
        g2 = self.box_g2(self.anchors(piece, torch.full_like(piece[:, 0], self.v.s_end)))
        t_min = torch.searchsorted(self.X ** 2 + 1e-6, g2.contiguous(), right=False)
        hi = piece[:, self.ai["t"]] + self.cont_top[piece[:, self.ai["len"]].clamp(0, len(self.cont_top) - 1)]
        if self.on["t_cap"]:
            hi = torch.clamp(hi, max=self.t_cap)
        return t_min <= hi

    def X2(self, t):
        return self.X[t.clamp(0, self.v.n_t - 1)] ** 2 + 1e-6

    def ptr_cone(self, t, g2min, valid):
        ok = (g2min <= self.X2(t)[:, None]) & valid
        return torch.cat([torch.ones_like(ok[:, :1]), ok], 1)

    def s_cone(self, t, g2_row):
        return g2_row <= self.X2(t)[:, None]

    def d_cone(self, name, n, t, anc, hx=None, hy=None):
        g = self.v.g
        dv = torch.as_tensor(self.v.dec_d(np.arange(n)), device=anc.device)[None, :]
        X2 = self.X2(t)[:, None]
        axis = "xyz".index(name[1])
        c = (anc[:, axis:axis + 1] + dv).clamp(0, g - 1)
        if name == "dx":
            return self._g(c, "xy") ** 2 + self._axis_min(anc, 1) ** 2 + self._axis_min(anc, 2) ** 2 <= X2
        gx2 = self._g(hx, "xy")[:, None] ** 2
        if name == "dy":
            return gx2 + self._g(c, "xy") ** 2 + self._axis_min(anc, 2) ** 2 <= X2
        return gx2 + self._g(hy, "xy")[:, None] ** 2 + self._g(c, "z") ** 2 <= X2

    def support(self, name, n, hung):
        N = hung.shape[0]
        ok = torch.ones(N, n, dtype=torch.bool, device=hung.device)
        if not self.on["support"]:
            return ok
        sup = self.s_ok if name == "s" else self.d_ok
        return torch.where(hung[:, None], sup[None, :n].expand(N, n), ok)

    def lt(self, n_lt, is_cont, prev_lt, t, length, tok=None):
        N = t.shape[0]
        b = torch.arange(n_lt, device=t.device)[None, :]
        ok = torch.ones(N, n_lt, dtype=torch.bool, device=t.device)
        if not self.on["lt"]:
            return ok
        ok &= ~is_cont[:, None] | (b <= prev_lt[:, None])
        end = length < LT_LEN_MIN
        if self.t_rule is not None and self.on["t_cap"]:
            end |= t + self.t_rule[length.clamp(0, len(self.t_rule) - 1)] - self.t_win_lo > self.t_cap
        if tok is not None and self.on["hung_cone"] and self.on["cont_window"] and self.cont:
            end |= ~self.cont_reachable(tok)
        return ok & (~end[:, None] | (b == 0))

    def lc(self, n_c, lq):
        N = lq.shape[0]
        if not self.on["q_per_record"]:
            return torch.ones(N, n_c, dtype=torch.bool, device=lq.device)
        return self.lc_ok[lq.clamp(0, self.v.n_q - 1)]


def rel_to_abs(rel_logits, floor):
    n = rel_logits.shape[1]
    b = torch.arange(n, device=rel_logits.device)[None, :]
    idx = (b - floor[:, None]).clamp(0, n - 1)
    out = rel_logits.gather(1, idx)
    return out.masked_fill(b < floor[:, None], float("-inf"))


def apply_mask(logits, ok, target=None, count=None, key=None, active=None):
    if target is not None:
        tg_ok = ok.gather(1, target.clamp(0, ok.shape[1] - 1)[:, None])[:, 0]
        if count is not None:
            bad = ~tg_ok if active is None else (~tg_ok & active)
            count[key] = count.get(key, 0) + int(bad.sum())
        ok = ok | ~tg_ok[:, None]
    else:
        none = ~ok.any(1)
        if none.any():
            SAMPLE_FALLBACKS[key] = SAMPLE_FALLBACKS.get(key, 0) + int(none.sum())
            ok = ok.clone()
            ok[none, logits[none].argmax(1)] = True
    return logits.masked_fill(~ok, float("-inf"))


def check_order(model):
    o = {model.names[a]: p for p, a in enumerate(model.order)}
    need = [("kind", "t"), ("t", "ix"), ("ix", "iy"), ("iy", "iz"), ("ptr", "s"), ("ptr", "dx"), ("ptr", "dy"), ("ptr", "dz"), ("lq", "lc")]
    if "lt" in o:
        need += [("t", "lt"), ("len", "lt")]
    R = getattr(model, "rules", None)
    if R is not None and R.on["hung_cone"]:
        need += [("t", "ptr"), ("s", "dx"), ("dx", "dy"), ("dy", "dz"), ("t", "s")]
    bad = [f"{a} before {b}" for a, b in need if o[a] > o[b]]
    if bad:
        raise ValueError("model.attr_order breaks the rule order: " + ", ".join(bad))


def constrain(R, model, name, lg, tok, ctx, target=None, count=None, active=None):
    ai = model.ai
    n = lg.shape[1]
    is_cont = tok[:, ai["kind"]] == KIND_CONT
    if name == "kind":
        ok = R.kind(n, ctx["prev_lt"], ctx["has_prev"], ctx["t_rule"], ctx.get("prev_tok") if not R.detail else None)
    elif name == "t":
        floor = R.t_floor(is_cont, ctx["t_order"], ctx["t_rule"])
        if target is not None:
            floor = torch.where(target >= floor, floor, torch.zeros_like(floor))
        if getattr(model, "t_rel", False):
            lg = rel_to_abs(lg, floor)
        ok = R.t(n, is_cont, ctx["t_order"], ctx["t_rule"], ctx["prev_t"], ctx["prev_len"])
        if R.on["hung_cone"] and R.cont and not R.detail and "prev_tok" in ctx:

            if "_cont_g2" not in ctx:
                pt = ctx["prev_tok"]
                ctx["_cont_g2"] = R.box_g2(R.anchors(pt, torch.full_like(pt[:, 0], R.v.s_end)))
            b = torch.arange(n, device=lg.device)
            ok = ok & (~is_cont[:, None] | (R.X[b.clamp(max=R.v.n_t - 1)][None, :] ** 2 + 1e-6 >= ctx["_cont_g2"][:, None]))
    elif name in ("ix", "iy", "iz"):
        ok = R.position(name, n, tok[:, ai["t"]], tok[:, ai["ix"]], tok[:, ai["iy"]])


        free = ((tok[:, ai["kind"]] == KIND_STROKE) | is_cont) & (tok[:, ai["ptr"]] == 0)
        ok = ok | ~free[:, None]
    elif name in ("s", "dx", "dy", "dz"):
        ok = R.support(name, n, tok[:, ai["ptr"]] > 0)
        if R.on["hung_cone"]:
            ok = ok & _hung_cone(R, model, name, n, tok, ctx, is_cont)
    elif name == "lt":
        ok = R.lt(n, is_cont, ctx["prev_lt"], tok[:, ai["t"]], tok[:, ai["len"]], tok)
    elif name == "lc":
        ok = R.lc(n, tok[:, ai["lq"]])
    else:
        return lg
    return apply_mask(lg, ok, target, count, name, active)


def _parent(R, model, tok, ctx):
    if "skel" in ctx:
        K = ctx["skel"].shape[1]
        return ctx["skel"][ctx["row_b"], (tok[:, model.ai["ptr"]] - 1).clamp(0, K - 1)]
    return ctx["prev_tok"]


def _hung_cone(R, model, name, n, tok, ctx, is_cont):
    ai = model.ai
    N = tok.shape[0]
    every = torch.ones(N, n, dtype=torch.bool, device=tok.device)
    if R.detail:
        if "skel" not in ctx:
            return every
        hung = ((tok[:, ai["kind"]] == KIND_STROKE) | is_cont) & (tok[:, ai["ptr"]] > 0)
    else:
        if "prev_tok" not in ctx:
            return every
        hung = is_cont
    t = tok[:, ai["t"]]
    if name == "s":
        if not R.detail:
            return every
        K = ctx["skel"].shape[1]
        g2 = ctx["piece_g2"][ctx["row_b"], (tok[:, ai["ptr"]] - 1).clamp(0, K - 1)]
        return every & (~hung[:, None] | R.s_cone(t, g2))
    if name == "dx" or "_anc" not in ctx:
        ctx["_anc"] = R.anchors(_parent(R, model, tok, ctx), tok[:, ai["s"]])
    anc = ctx["_anc"]
    hx = (anc[:, 0] + torch.as_tensor(R.v.dec_d(tok[:, ai["dx"]].cpu().numpy()), device=tok.device)).clamp(0, R.v.g - 1)
    hy = (anc[:, 1] + torch.as_tensor(R.v.dec_d(tok[:, ai["dy"]].cpu().numpy()), device=tok.device)).clamp(0, R.v.g - 1)
    return every & (~hung[:, None] | R.d_cone(name, n, t, anc, hx, hy))


def constrain_ptr(R, model, pl, tok, ctx, target=None, count=None, active=None):
    if R is None or not R.on["hung_cone"] or "piece_g2min" not in ctx:
        return pl
    ok = R.ptr_cone(tok[:, model.ai["t"]], ctx["piece_g2min"][ctx["row_b"]], ctx["skel_valid"][ctx["row_b"]])
    return apply_mask(pl, ok[:, : pl.shape[1]], target, count, "ptr", active)


def hung_context(R, skel, skel_valid, rows_per_b):
    if R is None or not R.on["hung_cone"]:
        return {}
    B, K, A = skel.shape
    g2 = R.piece_g2(skel.reshape(B * K, A)).reshape(B, K, -1)
    g2 = torch.where(skel_valid[..., None], g2, torch.full_like(g2, float("inf")))
    return dict(skel=skel, skel_valid=skel_valid, row_b=torch.arange(B, device=skel.device).repeat_interleave(rows_per_b),
                piece_g2=g2, piece_g2min=g2.min(2).values)


def train_context(model, tok_in, tstart=None):
    ai = model.ai
    B, n, A = tok_in.shape
    dev = tok_in.device
    prev = torch.cat([torch.zeros(B, 1, A, dtype=tok_in.dtype, device=dev), tok_in], 1)
    has_prev = torch.zeros(B, n + 1, dtype=torch.bool, device=dev)
    has_prev[:, 1:] = tok_in[..., ai["kind"]] > 0
    kin = tok_in[..., ai["kind"]]
    if model.rules.detail:
        t_order = prev[..., ai["t"]].clone()
        t_order[:, 0] = tstart if tstart is not None else 0
    else:
        head = (kin == KIND_STROKE)
        ht = torch.where(head, tok_in[..., ai["t"]], torch.zeros_like(kin))
        cm = torch.cummax(ht, 1).values if n > 0 else ht
        t_order = torch.cat([torch.zeros(B, 1, dtype=tok_in.dtype, device=dev), cm], 1)
    prev_lt = prev[..., ai["lt"]] if "lt" in ai else torch.zeros(B, n + 1, dtype=tok_in.dtype, device=dev)
    prev_lt = torch.where(has_prev, prev_lt, torch.zeros_like(prev_lt))
    t_rule = model.rule_t(prev) if getattr(model, "cont", False) else torch.zeros(B, n + 1, dtype=tok_in.dtype, device=dev)
    f = lambda x: x.reshape(-1)
    return dict(has_prev=f(has_prev), prev_lt=f(prev_lt), prev_t=f(prev[..., ai["t"]]), prev_len=f(prev[..., ai["len"]]), t_order=f(t_order), t_rule=f(t_rule),
                prev_tok=prev.reshape(-1, A))


def sample_context(model, prev, t_order, B, device):
    z = torch.zeros(B, dtype=torch.long, device=device)
    if prev is None:
        return dict(has_prev=torch.zeros(B, dtype=torch.bool, device=device), prev_lt=z, prev_t=z, prev_len=z, t_order=t_order, t_rule=z,
                    prev_tok=torch.zeros(B, model.A, dtype=torch.long, device=device))
    ai = model.ai
    has = prev[:, ai["kind"]] > 0
    lt = prev[:, ai["lt"]] if "lt" in ai else z
    return dict(has_prev=has, prev_lt=torch.where(has, lt, z), prev_t=prev[:, ai["t"]], prev_len=prev[:, ai["len"]], t_order=t_order,
                t_rule=model.rule_t(prev) if getattr(model, "cont", False) else z, prev_tok=prev)
