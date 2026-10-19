"""Detail model: strokes conditioned on the skeleton."""
from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .data import budget_bins, enc_cum
from .model import N_PREFIX, ShowerLM, sample_logits
from .tokens import KIND_CONT, KIND_EOS, KIND_PAD, KIND_STROKE

TYPE_PREFIX, TYPE_DETAIL, TYPE_SKEL = 0, 1, 2


def detail_mask(K, n_prefix, n, window, skel_len):
    dev = skel_len.device
    L = K + n_prefix + n
    i = torch.arange(L, device=dev)[:, None]
    j = torch.arange(L, device=dev)[None, :]
    sq, sk = i < K, j < K
    m = sq & sk & (j <= i)
    m |= ~sq & sk
    ii, jj = i - K, j - K
    band = ((jj <= ii) & (jj > ii - window)) | ((jj < n_prefix) & (jj <= ii))
    m |= ~sq & ~sk & band
    valid = torch.arange(K, device=dev)[None, :] < skel_len[:, None]
    col = torch.ones(len(skel_len), L, dtype=torch.bool, device=dev)
    col[:, :K] = valid
    M = m[None] & col[:, None, :]
    padq = torch.zeros(len(skel_len), L, dtype=torch.bool, device=dev)
    padq[:, :K] = ~valid
    M |= padq[:, :, None] & torch.eye(L, dtype=torch.bool, device=dev)[None]
    return M[:, None]


class DetailLM(ShowerLM):
    def __init__(self, *args, k_max=1024, **kw):
        super().__init__(*args, **kw)
        self.k_max = int(k_max)
        self.type_emb = nn.Embedding(3, self.d)
        self.par_proj = nn.Linear(self.d, self.d)
        self.ptr_in = nn.Linear(self.d, self.heads[0].in_features)
        for mod in (self.type_emb, self.par_proj, self.ptr_in):
            mod.apply(self._init)
        o = {self.names[a]: p for p, a in enumerate(self.order)}
        assert o["ptr"] < min(o["s"], o["dx"], o["dy"], o["dz"]) and max(o["s"], o["dx"], o["dy"], o["dz"]) < min(o["ix"], o["iy"], o["iz"]),\
            "attr_order must decode ptr, then s / dx / dy / dz, then ix / iy / iz"
        self.lt_index = self.ai.get("lt")


    def embed_skel(self, skel):
        return sum(self.emb[a](skel[..., a]) for a in range(self.A)) + self.type_emb.weight[TYPE_SKEL]

    def embed_detail(self, tok, e_skel, bud=None):
        ip = self.ai["ptr"]
        ptr = tok[..., ip]
        x = sum(self.emb[a]((ptr > 0).long() if a == ip else tok[..., a]) for a in range(self.A))
        K = e_skel.shape[1]
        if K > 0:
            idx = (ptr - 1).clamp(0, K - 1)
            par = torch.gather(e_skel, 1, idx[..., None].expand(-1, -1, e_skel.shape[2]))
            x = x + self.par_proj(par) * (ptr > 0)[..., None].to(par.dtype)
        if self.budget and bud is not None:
            x = x + self.bud_n(bud[..., 0]) + self.bud_q(bud[..., 1])
        return x + self.type_emb.weight[TYPE_DETAIL]

    def ptr_feature(self, h_skel, ptr):
        none = self.attr_in[self.ai["ptr"]](torch.zeros_like(ptr))
        K = h_skel.shape[1]
        if K == 0:
            return none
        idx = (ptr - 1).clamp(0, K - 1)
        f = self.ptr_in(torch.gather(h_skel, 1, idx[..., None].expand(-1, -1, h_skel.shape[2])))
        return torch.where((ptr > 0)[..., None], f, none)

    def attr_logits(self, h, target, ptr_feat=None):
        x0 = self.h_proj(h)[:, None, :]
        ip = self.ai["ptr"]
        xs = [(ptr_feat if (a == ip and ptr_feat is not None) else self.attr_in[a](target[:, a]))[:, None, :] for a in self.order[:-1]]
        x = torch.cat([x0] + xs, 1) + self.slot.weight[None, : self.A, :]
        for blk in self.attr_blocks:
            x = blk(x, None)
        x = self.attr_ln(x)
        x = x[:, torch.as_tensor([self.order.index(a) for a in range(self.A)], device=x.device)]
        return x, [self.heads[a](x[:, a]) for a in range(self.A)]

    def skel_pointer_logits(self, y_ptr, h_skel, skel_valid):
        q = self.ptr_q(y_ptr)
        none = (q @ self.ptr_none)[..., None] / math.sqrt(self.d)
        if h_skel.shape[1] == 0:
            return none
        s = torch.einsum("bid,bjd->bij", q, self.ptr_k(h_skel)) / math.sqrt(self.d)
        s = s.masked_fill(~skel_valid[:, None, :], float("-inf"))
        return torch.cat([none, s], -1)


    def _row_terms(self, skel, skel_len, prefix, tok_in, tok_out, bud=None):
        B, n, A = tok_in.shape
        K = skel.shape[1]
        e_skel = self.embed_skel(skel)
        x = torch.cat([e_skel, self.embed_prefix(prefix), self.embed_detail(tok_in, e_skel, bud)], 1)
        h = self.trunk(x, detail_mask(K, N_PREFIX, n, self.window, skel_len))
        h_skel = h[:, :K]
        ctx = h[:, K + N_PREFIX - 1: K + N_PREFIX + n]
        kind = tok_out[..., self.ai["kind"]]
        valid = kind > 0
        st = (kind == KIND_STROKE) | (kind == KIND_CONT)
        ptr = tok_out[..., self.ai["ptr"]]
        pf = self.ptr_feature(h_skel, ptr)
        N = B * (n + 1)
        tg = tok_out.reshape(N, A)
        y, logits = self.attr_logits(ctx.reshape(N, -1), tg, pf.reshape(N, -1))
        v, s_ = valid.reshape(-1), st.reshape(-1)
        has_ptr = (tg[:, self.ai["ptr"]] > 0) & s_
        R = self.rules if (self.rules is not None and self.rules.any) else None
        self.last_viol = {}
        if R is not None:
            from .rules import constrain, constrain_ptr, hung_context, train_context
            R.to(tok_in.device)
            rctx = train_context(self, tok_in, tstart=prefix[:, 5])
            rctx.update(hung_context(R, skel, torch.arange(K, device=skel.device)[None, :] < skel_len[:, None], n + 1))
        terms = {}
        for a, name in enumerate(self.names):
            if name in ("ptr", "lt"):
                continue
            if name == "kind":
                m = v
            elif name in ("s", "dx", "dy", "dz"):
                m = has_ptr
            elif name in ("ix", "iy", "iz"):
                m = s_ & ~has_ptr
            else:
                m = s_
            lg = logits[a].float()
            if R is not None:
                lg = constrain(R, self, name, lg, tg, rctx, target=tg[:, a], count=self.last_viol, active=m)
            terms[name] = (F.cross_entropy(lg, tg[:, a], reduction="none"), m)
        skel_valid = torch.arange(K, device=skel.device)[None, :] < skel_len[:, None]
        pl = self.skel_pointer_logits(y[:, self.ai["ptr"]].reshape(B, n + 1, -1), h_skel, skel_valid).float().reshape(N, -1)
        if R is not None:
            pl = constrain_ptr(R, self, pl, tg, rctx, target=ptr.reshape(-1).clamp(0, K), count=self.last_viol, active=s_)
        terms["ptr"] = (F.cross_entropy(pl, ptr.reshape(-1).clamp(0, K), reduction="none"), s_)
        w = None
        if self.q_weight_pow > 0:
            lq = tg[:, self.ai["lq"]].float()
            q = torch.pow(10.0, (lq + 0.5) * 7.0 / self.sizes[self.ai["lq"]])
            w = torch.pow(q, self.q_weight_pow)
            w = w / w[s_].mean().clamp_min(1e-9) if s_.any() else w
        plan = (self.head_qtot(h[:, K]), self.head_ntok(h[:, K + 1]))
        return terms, plan, v, w

    def forward(self, skel, skel_len, prefix, tok_in, tok_out, bud=None, lm=None):
        terms, plan, v, w = self._row_terms(skel, skel_len, prefix, tok_in, tok_out, bud)
        out = {"loss_qtot": F.cross_entropy(plan[0].float(), prefix[:, 1]), "loss_ntok": F.cross_entropy(plan[1].float(), prefix[:, 2])}
        losses = {}
        for name, (ce, m) in terms.items():
            if not m.any():
                continue
            if w is not None and name in ("kind", "th", "ph", "len", "lq", "lc", "coh"):
                losses[name] = (ce[m] * w[m]).sum() / w[m].sum().clamp_min(1e-9)
            else:
                losses[name] = ce[m].mean()
        out.update({f"loss_{k}": val for k, val in losses.items()})
        out["loss"] = out["loss_qtot"] + out["loss_ntok"] + sum(losses.values())
        out["n_tokens"] = int(v.sum())
        out.update({f"viol_{k}": c for k, c in self.last_viol.items()})
        return out

    @torch.no_grad()
    def token_logprobs(self, skel, skel_len, prefix, tok_in, tok_out, bud=None):
        terms, _, v, _ = self._row_terms(skel, skel_len, prefix, tok_in, tok_out, bud)
        tot = torch.zeros_like(terms["kind"][0])
        for ce, m in terms.values():
            tot = tot - torch.where(m, ce, torch.zeros_like(ce))
        return tot.reshape(tok_out.shape[0], -1)


@torch.no_grad()
def decode_detail(model, ctx, h_skel, skel_valid, skels, vocab, temperature=1.0, rng=None, suppress_eos=False, attr_temp=None, forced=None, t_prev=None, rctx=None,
                  t_hi=None, t_lo=None):
    from .sample import cone_iz_min, masked_or_cap
    if rctx is not None:
        from .rules import constrain, constrain_ptr, hung_context
        t_prev = None
        if model.rules.on["hung_cone"] and "skel" not in rctx:
            rctx.update(hung_skel_context(model, skels, skel_valid, ctx.device))
    B = ctx.shape[0]
    dev = ctx.device
    ix = model.ai
    tok = torch.zeros(B, model.A, dtype=torch.long, device=dev)
    logp = torch.zeros(B, device=dev)
    x0 = model.h_proj(ctx)[:, None, :]
    inputs = []
    ptr = torch.zeros(B, dtype=torch.long, device=dev)
    pos_fixed = None
    for p, a in enumerate(model.order):
        x = torch.cat([x0] + inputs, 1) + model.slot.weight[None, : p + 1, :]
        for blk in model.attr_blocks:
            x = blk(x, None)
        y = model.attr_ln(x[:, p])
        name = model.names[a]
        ta = (attr_temp or {}).get(name, temperature)
        st = (tok[:, ix["kind"]] == KIND_STROKE) | (tok[:, ix["kind"]] == KIND_CONT)
        if name == "ptr":
            lg = model.skel_pointer_logits(y[:, None, :], h_skel, skel_valid)[:, 0].float()
            if rctx is not None:
                lg = constrain_ptr(model.rules, model, lg, tok, rctx)
            c = forced[:, a] if forced is not None else sample_logits(lg, ta, rng)
            logp = logp + torch.where(st, torch.log_softmax(lg, -1).gather(1, c.clamp(0, lg.shape[1] - 1)[:, None])[:, 0], torch.zeros_like(logp))
            tok[:, a] = c; ptr = c
            inputs.append(model.ptr_feature(h_skel, ptr[:, None]))
            continue
        lg = model.heads[a](y).float()
        if name == "lt":
            val = torch.zeros(B, dtype=torch.long, device=dev) if forced is None else forced[:, a]
            tok[:, a] = val
            inputs.append(model.attr_in[a](val)[:, None, :])
            continue
        if rctx is not None:
            lg = constrain(model.rules, model, name, lg, tok, rctx)
            if name == "kind" and forced is None and suppress_eos:
                lg[:, KIND_EOS] = float("-inf")
        if t_hi is not None and forced is None and name == "t":
            b = torch.arange(lg.shape[1], device=dev)[None, :]
            ok = (b <= t_hi[:, None]) & torch.isfinite(lg)
            lowest = torch.where(torch.isfinite(lg), b.expand_as(lg), torch.full_like(b.expand_as(lg), lg.shape[1])).min(1).values
            lg = masked_or_cap(lg, ok, torch.clamp(lowest, max=lg.shape[1] - 1))
        if t_lo is not None and forced is None and name == "t":
            b = torch.arange(lg.shape[1], device=dev)[None, :]
            fin = torch.isfinite(lg)
            highest = torch.where(fin, b.expand_as(lg), torch.full_like(b.expand_as(lg), -1)).max(1).values.clamp_min(0)
            lg = masked_or_cap(lg, (b >= t_lo[:, None]) & fin, highest)
        if getattr(model, "strict", None) is not None and forced is None and name in ("oz", "len"):
            lg = model.strict.apply(name, lg, tok, ix)
        if t_prev is not None and forced is None and name == "t":
            t_cap = int(vocab.enc_t(12.0 - 1e-9))
            b = torch.arange(lg.shape[1], device=dev)[None, :]
            lg = masked_or_cap(lg, (b <= t_cap) & (b >= t_prev[:, None]), torch.full_like(t_prev, t_cap))
        if t_prev is not None and forced is None and name == "iz":
            b = torch.arange(lg.shape[1], device=dev)[None, :]
            lg = masked_or_cap(lg, b >= cone_iz_min(vocab, tok[:, ix["t"]])[:, None], torch.full_like(t_prev, lg.shape[1] - 1))
        if name == "kind" and forced is None and rctx is None:
            lg[:, KIND_PAD] = float("-inf")
            if lg.shape[1] > KIND_CONT:
                lg[:, KIND_CONT] = float("-inf")
            if suppress_eos:
                lg[:, KIND_EOS] = float("-inf")
        hp = ptr > 0
        if forced is None and rctx is None and name in ("s", "dx", "dy", "dz"):


            lg = torch.where(hp[:, None], lg.masked_fill(~support_mask(model, vocab, name, lg.device)[None], float("-inf")), lg)
        val = forced[:, a] if forced is not None else sample_logits(lg, ta, rng)
        if name in ("s", "dx", "dy", "dz"):
            active = st & hp
            if forced is None:
                val = torch.where(hp, val, torch.zeros_like(val))
        elif name in ("ix", "iy", "iz"):
            active = st & ~hp
            if forced is None:
                if pos_fixed is None:
                    pos_fixed = torch.from_numpy(fixed_positions(tok.cpu().numpy(), skels, vocab)).to(dev)
                val = torch.where(hp, pos_fixed[:, "xyz".index(name[1])], val)
        elif name == "kind":
            active = torch.ones(B, dtype=torch.bool, device=dev)
        else:
            active = st
        logp = logp + torch.where(active, torch.log_softmax(lg, -1).gather(1, val[:, None])[:, 0], torch.zeros_like(logp))
        tok[:, a] = val
        inputs.append(model.attr_in[a](val)[:, None, :])
    return tok, logp


_SUPPORT = {}
PACE_F_MAX = 0.9


def hung_skel_context(model, skels, skel_valid, device):
    from .rules import hung_context
    B, K = skel_valid.shape
    skel = torch.zeros(B, K, model.A, dtype=torch.long, device=device)
    for b, sk in enumerate(skels):
        if len(sk):
            skel[b, : len(sk)] = torch.as_tensor(np.asarray(sk, dtype=np.int64)[:K], device=device)
    model.rules.to(device)
    return hung_context(model.rules, skel, skel_valid, 1)


def support_mask(model, vocab, name, device):
    key = (name, str(device))
    if key not in _SUPPORT:
        n = model.sizes[model.ai[name]]
        b = np.arange(n)
        ok = b >= 1 if name == "s" else (b >= 1) & (np.abs(vocab.dec_d(b)) <= 8)
        _SUPPORT[key] = torch.from_numpy(ok).to(device)
    return _SUPPORT[key]


def fixed_positions(tok, skels, vocab):
    ix = vocab.index
    out = np.zeros((len(tok), 3), np.int64)
    for b in range(len(tok)):
        p = int(tok[b, ix["ptr"]])
        if p > 0 and p <= len(skels[b]):
            a = vocab.anchor(skels[b][p - 1], int(tok[b, ix["s"]]))
            d = np.array([vocab.dec_d(int(tok[b, ix[c]])) for c in ("dx", "dy", "dz")]).reshape(3)
            out[b] = np.clip(a + d, 0, vocab.g - 1)
    return out


class DetailSampler:

    def __init__(self, model, vocab, skels, device, chunk=1024, rebuild_every=256, tstart_init=0):
        if not 0 < rebuild_every < chunk:
            raise ValueError("need 0 < rebuild_every < chunk")
        if chunk > model.window:
            raise ValueError("chunk must be <= the attention window")
        self.model, self.vocab, self.dev = model, vocab, device
        self.skels = [np.asarray(s, dtype=np.int64).reshape(-1, len(vocab.names)) for s in skels]
        self.B = B = len(skels)
        self.K = K = max(1, max(len(s) for s in self.skels))
        self.chunk, self.rebuild_every, self.tstart_init = int(chunk), int(rebuild_every), int(tstart_init)
        skel = torch.zeros(B, K, model.A, dtype=torch.long, device=device)
        for b, sk in enumerate(self.skels):
            if len(sk):
                skel[b, : len(sk)] = torch.from_numpy(sk).to(device)
        self.skel_len = torch.tensor([len(sk) for sk in self.skels], device=device)
        self.skel_valid = torch.arange(K, device=device)[None, :] < self.skel_len[:, None]
        self.e_skel = model.embed_skel(skel)
        self.skel_cache = [dict() for _ in model.blocks]
        self.h_skel = model.trunk(self.e_skel, detail_mask(K, 0, 0, model.window, self.skel_len), self.skel_cache)
        self.prefix = torch.zeros(B, N_PREFIX, dtype=torch.long, device=device)
        self.prefix[:, 5] = self.tstart_init
        self.hist_tok, self.hist_bud = [], []
        self.j, self.w0 = 0, 0
        self.n_rebuilds = 0

    def _run(self, x, n_prefix_new, upto_prefix=N_PREFIX):
        m = self.model
        cc = [dict(k=c["k"], v=c["v"]) for c in self.skel_cache]
        L = x.shape[1]
        i = torch.arange(L, device=self.dev)[:, None]; j = torch.arange(L, device=self.dev)[None, :]
        band = ((j <= i) & (j > i - m.window)) | ((j < n_prefix_new) & (j <= i))
        vis = torch.cat([self.skel_valid[:, None, :].expand(self.B, L, self.K), band[None].expand(self.B, L, L)], 2)
        h = m.trunk(x, vis[:, None], cc)
        return h, cc

    def run_prefix(self, upto=N_PREFIX):
        return self._run(self.model.embed_prefix(self.prefix)[:, :upto], upto)

    def plan(self, temperature=1.0, rng=None, forced_prefix=None):
        from .data import dec_ntok, dec_qtot
        m = self.model
        if forced_prefix is not None:
            self.prefix = forced_prefix.clone().to(self.dev)
        h, _ = self.run_prefix(1)
        lq = m.head_qtot(h[:, 0]).float()
        if forced_prefix is None:
            self.prefix[:, 1] = sample_logits(lq, temperature, rng)
        h, _ = self.run_prefix(2)
        ln = m.head_ntok(h[:, 1]).float()
        if forced_prefix is None:
            self.prefix[:, 2] = sample_logits(ln, temperature, rng)
        self.plan_logp = torch.log_softmax(lq, -1).gather(1, self.prefix[:, 1:2])[:, 0] + torch.log_softmax(ln, -1).gather(1, self.prefix[:, 2:3])[:, 0]
        self.q_plan = dec_qtot(self.prefix[:, 1].cpu().numpy()); self.n_plan = dec_ntok(self.prefix[:, 2].cpu().numpy())
        h, self.caches = self.run_prefix(N_PREFIX)
        self.ctx = h[:, -1]
        return self.q_plan, self.n_plan

    def feed(self, tok, bud=None):
        m = self.model
        self.hist_tok.append(tok); self.hist_bud.append(bud)
        L0 = self.caches[0]["k"].shape[2]
        n_in = self.j - self.w0
        col_valid = torch.cat([self.skel_valid, torch.ones(self.B, L0 - self.K + 1, dtype=torch.bool, device=self.dev)], 1)

        if n_in + 1 > m.window:
            col_valid[:, self.K + N_PREFIX: self.K + N_PREFIX + n_in + 1 - m.window] = False
        h = m.trunk(m.embed_detail(tok[:, None, :], self.e_skel, bud), col_valid[:, None, None, :], self.caches)
        self.ctx = h[:, -1]
        self.j += 1
        if self.j - self.w0 >= self.chunk:
            self.rebuild(self.j - (self.chunk - self.rebuild_every))

    def chunk_prefix(self, w0):
        v = self.vocab
        pre = self.prefix.clone()
        if w0 == 0:
            pre[:, 3] = 0; pre[:, 4] = 0; pre[:, 5] = self.tstart_init
            return pre
        toks = torch.stack(self.hist_tok[:w0], 1).cpu().numpy()
        q_before = v.dec_log(toks[..., v.index["lq"]], v.n_q, v.q_log_max).sum(1)
        for b in range(self.B):
            pre[b, 3] = enc_cum(min(q_before[b] / max(self.q_plan[b], 1.0), 0.999))
            pre[b, 4] = enc_cum(min(w0 / max(self.n_plan[b], 1.0), 0.999))
            pre[b, 5] = int(toks[b, w0 - 1, v.index["t"]])
        return pre

    def rebuild(self, w0):
        m = self.model
        self.prefix = self.chunk_prefix(w0)
        toks = torch.stack(self.hist_tok[w0: self.j], 1)
        bud = torch.cat(self.hist_bud[w0: self.j], 1) if (self.hist_bud and self.hist_bud[0] is not None) else None
        x = torch.cat([m.embed_prefix(self.prefix), m.embed_detail(toks, self.e_skel, bud)], 1)
        h, self.caches = self._run(x, N_PREFIX)
        self.ctx = h[:, -1]
        self.w0 = w0
        self.n_rebuilds += 1


@torch.no_grad()
def generate_detail(model, vocab, skels, device, temperature=1.0, max_tokens=20000, chunk=1024, rebuild_every=256, seed=0, verbose=False, min_tokens=0,
                    attr_temp=None, tstart_init=0, stop_at_plan="off", constraints=False, time_order=False, plan_model=None,
                    t_jump_max=None, t_first_max=None, pace=None):
    from .common import log
    from .sample import check_rule_order
    use_rules = getattr(model, "rules", None) is not None and model.rules.any
    if use_rules:
        from .rules import sample_context
        model.rules.to(device)
        constraints = False
    if constraints:
        check_rule_order(model)
    rng = torch.Generator(device=device).manual_seed(seed)
    ix = vocab.index
    S = DetailSampler(model, vocab, skels, device, chunk, rebuild_every, tstart_init)
    B = S.B
    if plan_model is None:
        q_plan, n_plan = S.plan(temperature, rng)
    else:
        Sp = DetailSampler(plan_model, vocab, skels, device, chunk, rebuild_every, tstart_init)
        Sp.plan(temperature, rng)
        q_plan, n_plan = S.plan(forced_prefix=Sp.prefix)
        del Sp
    toks = [[] for _ in range(B)]
    alive = torch.ones(B, dtype=torch.bool, device=device)
    q_sum = np.zeros(B)
    stop = [None] * B
    use_bud = bool(getattr(model, "budget", False))
    t_prev = torch.full((B,), int(tstart_init), dtype=torch.long, device=device)
    hctx = hung_skel_context(model, S.skels, S.skel_valid, device) if use_rules and model.rules.on["hung_cone"] else {}
    big = vocab.n_t
    for step in range(max_tokens):
        t_hi = None
        if t_jump_max is not None or t_first_max is not None:
            first = torch.tensor([len(t) == 0 for t in toks], dtype=torch.bool, device=device)
            nxt = t_prev + (int(t_jump_max) if t_jump_max is not None else big)
            t_hi = torch.where(first, torch.full_like(t_prev, int(t_first_max) if t_first_max is not None else big), nxt)
        t_lo = None
        if pace is not None:
            fr = np.array([len(toks[b]) / max(float(n_plan[b]), 1.0) for b in range(B)])
            lo = np.where(fr <= PACE_F_MAX, np.floor(np.interp(np.clip(fr, 0, 1), pace[0], pace[1])), 0)
            t_lo = torch.as_tensor(lo, dtype=torch.long, device=device)
            if t_hi is not None:
                t_lo = torch.minimum(t_lo, t_hi)
        tok, _ = decode_detail(model, S.ctx, S.h_skel, S.skel_valid, S.skels, vocab, temperature, rng, suppress_eos=step < min_tokens, attr_temp=attr_temp,
                               t_prev=(t_prev if time_order else torch.zeros_like(t_prev)) if constraints else None,
                               rctx=dict(sample_context(model, None, t_prev, B, device), **hctx) if use_rules else None, t_hi=t_hi, t_lo=t_lo)
        kind = tok[:, ix["kind"]]
        done_now = alive & (kind == KIND_EOS)
        alive = alive & (kind == KIND_STROKE)
        tk = tok.cpu().numpy()
        for b in range(B):
            if done_now[b]:
                stop[b] = "eos"
            if alive[b]:
                toks[b].append(tk[b]); q_sum[b] += float(vocab.dec_log(tk[b, ix["lq"]], vocab.n_q, vocab.q_log_max))
                if (stop_at_plan == "q" and q_sum[b] >= q_plan[b]) or (stop_at_plan == "n" and len(toks[b]) >= n_plan[b]):
                    alive[b] = False; stop[b] = "plan"
        if not alive.any():
            break
        tok = torch.where(alive[:, None], tok, torch.zeros_like(tok))
        tok[:, ix["kind"]] = torch.where(alive, torch.full_like(kind, KIND_STROKE), kind)
        t_prev = torch.where(alive, tok[:, ix["t"]], t_prev)
        bud = None
        if use_bud:
            rem_n = np.array([n_plan[b] - len(toks[b]) for b in range(B)]); rem_q = q_plan - q_sum
            bud = torch.from_numpy(budget_bins(rem_n, rem_q)).to(device)[:, None, :]
        S.feed(tok, bud)
        if verbose and (step + 1) % 1000 == 0:
            log(f"  detail step {step + 1}: alive {int(alive.sum())}/{B}, photons so far {q_sum.mean():.3g} (plan {q_plan.mean():.3g})")
    stop = [r if r is not None else "max" for r in stop]
    return [np.array(t, dtype=np.int64).reshape(-1, len(vocab.names)) for t in toks], dict(q_plan=q_plan, n_plan=n_plan, stop=stop, rebuilds=S.n_rebuilds)
