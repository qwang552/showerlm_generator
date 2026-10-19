"""Transformer that predicts one stroke (20 attributes) at a time."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

PREFIX = ["bos", "qtot", "ntok", "qcum", "ncum", "tstart"]
N_PREFIX = len(PREFIX)


def banded_causal_mask(L, window, n_prefix, device):
    i = torch.arange(L, device=device)[:, None]
    j = torch.arange(L, device=device)[None, :]
    m = (j <= i) & (j > i - window)
    m |= (j < n_prefix) & (j <= i)
    return m


def landmark_mask(L, window, n_prefix, lm):
    B = lm.shape[0]
    m = banded_causal_mask(L, window, n_prefix, lm.device)[None].expand(B, L, L).clone()
    i = torch.arange(L, device=lm.device)
    col = torch.zeros(B, L, dtype=torch.bool, device=lm.device)
    col[:, n_prefix:] = lm
    m |= col[:, None, :] & (i[None, :] <= i[:, None])[None]
    return m[:, None]


class Block(nn.Module):
    def __init__(self, d, n_heads, dropout=0.0):
        super().__init__()
        self.h = n_heads
        self.ln1 = nn.LayerNorm(d); self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d); self.proj = nn.Linear(d, d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)
        self.dropout = dropout

    def forward(self, x, mask=None, cache=None):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q = q.view(B, L, self.h, -1).transpose(1, 2); k = k.view(B, L, self.h, -1).transpose(1, 2); v = v.view(B, L, self.h, -1).transpose(1, 2)
        if cache is not None:
            if "k" in cache:
                k = torch.cat([cache["k"], k], 2); v = torch.cat([cache["v"], v], 2)
            cache["k"], cache["v"] = k, v


            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None and k.shape[2] == L and L > 1)
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=mask is None, dropout_p=self.dropout if self.training else 0.0)
        y = y.transpose(1, 2).reshape(B, L, d)
        x = x + self.drop(self.proj(y))
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x


class ShowerLM(nn.Module):
    def __init__(self, vocab_sizes, attr_names, n_t, n_qtot=64, n_ntok=64, n_cum=33, d=512, n_layers=8, n_heads=8,
                 d_attr=256, n_attr_layers=2, n_attr_heads=4, window=2048, w_ptr=2048, dropout=0.0, attr_order=None, causal_ptr=False,
                 budget=False, n_bud=40, landmarks=False, t_window=0, t_rule_table=None, q_weight_pow=0.0, t_rel=False):
        super().__init__()
        self.sizes = list(vocab_sizes); self.names = list(attr_names); self.A = len(self.sizes)
        self.ai = {n: i for i, n in enumerate(self.names)}

        self.cont = self.sizes[0] >= 4
        self.budget, self.landmarks, self.q_weight_pow = bool(budget), bool(landmarks), float(q_weight_pow)


        self.rules, self.t_rel = None, bool(t_rel)

        lo, hi = (int(t_window), int(t_window)) if isinstance(t_window, (int, float)) else (int(t_window[0]), int(t_window[1]))
        self.t_win_lo, self.t_win_hi = max(lo, 0), max(hi, 0)
        self.t_window = max(self.t_win_lo, self.t_win_hi)
        if self.budget:
            self.bud_n = nn.Embedding(n_bud, d); self.bud_q = nn.Embedding(n_bud, d)

        self.register_buffer("t_rule", torch.as_tensor(list(t_rule_table) if t_rule_table is not None else [0] * self.sizes[self.ai["len"]], dtype=torch.long), persistent=False)

        names = list(attr_order) if attr_order else list(self.names)
        assert sorted(names) == sorted(self.names) and names[0] == "kind", "attr_order must be a permutation of the attributes starting with kind"
        self.order = [self.ai[n] for n in names]
        self.causal_ptr = bool(causal_ptr)
        self.d, self.window, self.w_ptr = d, window, w_ptr
        self.prefix_sizes = dict(bos=1, qtot=n_qtot, ntok=n_ntok, qcum=n_cum, ncum=n_cum, tstart=n_t)
        self.emb = nn.ModuleList([nn.Embedding(n, d) for n in self.sizes])
        self.prefix_emb = nn.ModuleDict({k: nn.Embedding(n, d) for k, n in self.prefix_sizes.items()})
        self.type_emb = nn.Embedding(2, d)
        self.blocks = nn.ModuleList([Block(d, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d)
        self.head_qtot = nn.Linear(d, n_qtot); self.head_ntok = nn.Linear(d, n_ntok)

        self.h_proj = nn.Linear(d, d_attr)
        self.attr_in = nn.ModuleList([nn.Embedding(n, d_attr) for n in self.sizes])
        self.slot = nn.Embedding(self.A, d_attr)
        self.attr_blocks = nn.ModuleList([Block(d_attr, n_attr_heads, dropout) for _ in range(n_attr_layers)])
        self.attr_ln = nn.LayerNorm(d_attr)
        self.heads = nn.ModuleList([nn.Linear(d_attr, n) for n in self.sizes])
        self.ptr_q = nn.Linear(d_attr, d); self.ptr_k = nn.Linear(d, d)
        self.ptr_none = nn.Parameter(torch.zeros(d))
        self.apply(self._init)

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)


    def embed_strokes(self, tok, bud=None):
        x = sum(self.emb[a](tok[..., a]) for a in range(self.A))
        if self.budget and bud is not None:
            x = x + self.bud_n(bud[..., 0]) + self.bud_q(bud[..., 1])
        return x + self.type_emb.weight[1]

    def cont_t_mask(self, t_logits, t_rule):
        b = torch.arange(t_logits.shape[1], device=t_logits.device)[None, :]
        d = b - t_rule[:, None]
        inside = (d >= -self.t_win_lo) & (d <= self.t_win_hi)
        inside |= b == t_rule.clamp(0, t_logits.shape[1] - 1)[:, None]
        return t_logits.masked_fill(~inside, float("-inf"))

    def cont_t_target(self, t_target, t_rule):
        return torch.minimum(torch.maximum(t_target, t_rule - self.t_win_lo), t_rule + self.t_win_hi).clamp(0, self.sizes[self.ai["t"]] - 1)

    def rule_t(self, prev_tok):
        return prev_tok[..., self.ai["t"]] + self.t_rule[prev_tok[..., self.ai["len"]]]

    def embed_prefix(self, prefix):
        xs = [self.prefix_emb[k](prefix[:, i]) for i, k in enumerate(PREFIX)]
        return torch.stack(xs, 1) + self.type_emb.weight[0]

    def trunk(self, x, mask=None, caches=None):
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask, None if caches is None else caches[i])
        return self.ln_f(x)


    def attr_logits(self, h, target):
        N = h.shape[0]
        x0 = self.h_proj(h)[:, None, :]
        xs = [self.attr_in[a](target[:, a])[:, None, :] for a in self.order[:-1]]
        x = torch.cat([x0] + xs, 1) + self.slot.weight[None, : self.A, :]
        for blk in self.attr_blocks:
            x = blk(x, None)
        x = self.attr_ln(x)
        x = x[:, torch.as_tensor([self.order.index(a) for a in range(self.A)], device=x.device)]
        return x, [self.heads[a](x[:, a]) for a in range(self.A)]

    def pointer_logits(self, y_ptr, keys, i_idx, j_valid):
        q = self.ptr_q(y_ptr)
        s = torch.einsum("bid,bjd->bij", q, self.ptr_k(keys)) / math.sqrt(self.d)
        none = (q @ self.ptr_none)[..., None] / math.sqrt(self.d)
        s = s.masked_fill(~(j_valid if j_valid.dim() == 3 else j_valid[None]), float("-inf"))
        return torch.cat([none, s], -1)


    def forward(self, prefix, tok_in, tok_out, bud=None, lm=None):
        B, n, A = tok_in.shape
        x = torch.cat([self.embed_prefix(prefix), self.embed_strokes(tok_in, bud)], 1)
        L = x.shape[1]
        mask = landmark_mask(L, self.window, N_PREFIX, lm) if (self.landmarks and lm is not None) else banded_causal_mask(L, self.window, N_PREFIX, x.device)
        h = self.trunk(x, mask)
        out = {}
        out["loss_qtot"] = F.cross_entropy(self.head_qtot(h[:, 0]), prefix[:, 1])
        out["loss_ntok"] = F.cross_entropy(self.head_ntok(h[:, 1]), prefix[:, 2])
        ctx = h[:, N_PREFIX - 1: N_PREFIX + n]
        kind = tok_out[..., self.ai["kind"]]
        valid = kind > 0
        is_stroke = (kind == 1) | (kind == 3)
        is_cont = kind == 3
        hc = ctx.reshape(B * (n + 1), -1); tg = tok_out.reshape(B * (n + 1), A)
        y, logits = self.attr_logits(hc, tg)
        v = valid.reshape(-1); st = is_stroke.reshape(-1); co = is_cont.reshape(-1)
        has_ptr = (tg[:, self.ai["ptr"]] > 0) & st

        w = None
        if self.q_weight_pow > 0:
            lq = tg[:, self.ai["lq"]].float()
            q = torch.pow(10.0, (lq + 0.5) * 7.0 / self.sizes[self.ai["lq"]])
            w = torch.pow(q, self.q_weight_pow)
            w = w / w[st].mean().clamp_min(1e-9) if st.any() else w

        R = self.rules if (self.rules is not None and self.rules.any) else None
        viol = {}
        if R is not None:
            from .rules import constrain, train_context
            R.to(x.device)
            rctx = train_context(self, tok_in)

        t_rule = None
        if self.cont and self.t_window > 0 and R is None:
            prev = torch.cat([tok_in[:, :1], tok_in[:, :-1]], 1) if n > 0 else tok_in
            t_rule = torch.cat([self.rule_t(prev), self.rule_t(tok_in[:, -1:])], 1).reshape(-1) if n > 0 else torch.zeros(B, dtype=torch.long, device=x.device)
        losses = {}
        for a, name in enumerate(self.names):
            if name == "ptr":
                continue
            if name == "kind":
                m = v
            elif name == "s":
                m = has_ptr & ~co
            elif name in ("dx", "dy", "dz"):
                m = has_ptr
            elif name in ("ix", "iy", "iz"):
                m = st & ~has_ptr
            else:
                m = st
            if not m.any():
                continue
            lg = logits[a].float(); tga = tg[:, a]
            if R is not None:
                lg = constrain(R, self, name, lg, tg, rctx, target=tga, count=viol, active=m)
            if name == "t" and t_rule is not None and co.any():
                lg = torch.where(co[:, None], self.cont_t_mask(lg, t_rule), lg)
                tga = torch.where(co, self.cont_t_target(tga, t_rule), tga)
            if w is not None and name in ("kind", "th", "ph", "len", "lq", "lc", "coh", "lt"):
                ce = F.cross_entropy(lg[m], tga[m], reduction="none")
                losses[name] = (ce * w[m]).sum() / w[m].sum().clamp_min(1e-9)
            else:
                losses[name] = F.cross_entropy(lg[m], tga[m])

        keys = h[:, N_PREFIX: N_PREFIX + n]
        yp = y[:, self.ai["ptr"]].reshape(B, n + 1, -1)
        i_idx = torch.arange(n + 1, device=x.device)[:, None]; j_idx = torch.arange(n, device=x.device)[None, :]
        j_valid = (j_idx < i_idx) & (i_idx - j_idx <= self.w_ptr)
        if self.causal_ptr:
            it = self.ai["t"]
            j_valid = j_valid[None] & (tok_in[:, None, :, it] <= tok_out[:, :, None, it])
        pl = self.pointer_logits(yp, keys, i_idx, j_valid).float()
        ptr = tok_out[..., self.ai["ptr"]]
        tgt = torch.where(ptr > 0, i_idx[None, :, 0] - ptr + 1, torch.zeros_like(ptr))
        m = is_stroke & ~is_cont
        if m.any():
            losses["ptr"] = F.cross_entropy(pl[m], tgt[m])
        out.update({f"loss_{k}": val for k, val in losses.items()})
        out["loss"] = out["loss_qtot"] + out["loss_ntok"] + sum(losses.values())
        out["n_tokens"] = int(v.sum())
        out.update({f"viol_{k}": c for k, c in viol.items()})
        return out


    @torch.no_grad()
    def decode_attributes(self, h, fixed, temperature=1.0, keys=None, ptr_mask=None, rng=None):
        B = h.shape[0]
        tok = torch.zeros(B, self.A, dtype=torch.long, device=h.device)
        x0 = self.h_proj(h)[:, None, :]
        for a in range(self.A):
            xs = [x0] + [self.attr_in[b](tok[:, b])[:, None, :] for b in range(a)]
            x = torch.cat(xs, 1) + self.slot.weight[None, : a + 1, :]
            for blk in self.attr_blocks:
                x = blk(x, None)
            y = self.attr_ln(x[:, a])
            name = self.names[a]
            if name in fixed and not callable(fixed[name]):
                tok[:, a] = fixed[name]; continue
            if name in fixed:
                val = fixed[name](tok)
                if val is not None:
                    tok[:, a] = val; continue
            if name == "ptr":
                if keys is None or keys.shape[1] == 0:
                    tok[:, a] = 0; continue
                q = self.ptr_q(y)
                s = torch.einsum("bd,bjd->bj", q, self.ptr_k(keys)) / math.sqrt(self.d)
                s = s.masked_fill(~ptr_mask, float("-inf"))
                none = (q @ self.ptr_none)[:, None] / math.sqrt(self.d)
                logits = torch.cat([none, s], 1).float()
                c = sample_logits(logits, temperature, rng)
                M = keys.shape[1]
                tok[:, a] = torch.where(c > 0, M - (c - 1), torch.zeros_like(c))
            else:
                logits = self.heads[a](y).float()
                tok[:, a] = sample_logits(logits, temperature, rng)
        return tok


def sample_logits(logits, temperature=1.0, rng=None, top_p=None):
    if temperature <= 0:
        return logits.argmax(-1)
    p = F.softmax(logits / temperature, -1)
    if top_p is not None and top_p < 1.0:
        sp, si = torch.sort(p, descending=True, dim=-1)
        cum = torch.cumsum(sp, -1)
        keep = cum - sp < top_p
        sp = sp * keep
        p = torch.zeros_like(p).scatter_(-1, si, sp)
        p = p / p.sum(-1, keepdim=True)
    return torch.multinomial(p, 1, generator=rng).squeeze(-1)


def count_params(model):
    return sum(p.numel() for p in model.parameters())
