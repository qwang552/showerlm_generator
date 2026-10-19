"""Skeleton sampling and model loading."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .common import log
from .data import budget_bins, dec_ntok, dec_qtot, enc_cum
from .model import N_PREFIX, sample_logits
from .tokens import KIND_CONT, KIND_EOS, KIND_STROKE, Vocab
from .train import build_model


def load_run(run, device, checkpoint="best"):
    run = Path(run)
    ck_path = run / f"{checkpoint}.pt"
    if not ck_path.exists():
        ck_path = run / "last.pt"
    ck = torch.load(ck_path, map_location=device, weights_only=False)
    cfg = ck["config"]
    vocab = Vocab.from_dict(ck["vocab"])
    model = build_model(cfg, vocab, device)
    model.load_state_dict(ck["ema"])
    model.eval()
    return model, vocab, cfg, ck_path


@torch.no_grad()
def generate(model, vocab, B, device, temperature=1.0, max_tokens=20000, prefix_refresh=256, seed=0, verbose=False, min_tokens=0, stop_at_plan="off", attr_temp=None, tstart_init=0, causal_ptr=False,
             birth_ptr=True, constraints=False, time_order=False):
    use_rules = getattr(model, "rules", None) is not None and model.rules.any
    if use_rules:
        from .rules import sample_context
        model.rules.to(device)
        constraints = False
    if constraints:
        check_rule_order(model)
    rng = torch.Generator(device=device).manual_seed(seed)
    ix = vocab.index
    window = model.window
    cont = bool(getattr(model, "cont", False)) and vocab.v3
    use_bud = bool(getattr(model, "budget", False))
    use_lm = bool(getattr(model, "landmarks", False))
    lm_lq = int(getattr(model, "landmark_lq", -1))

    prefix = torch.zeros(B, N_PREFIX, dtype=torch.long, device=device)
    prefix[:, 5] = int(tstart_init)
    h = model.trunk(model.embed_prefix(prefix)[:, :1])
    prefix[:, 1] = sample_logits(model.head_qtot(h[:, 0]).float(), temperature, rng)
    h = model.trunk(model.embed_prefix(prefix)[:, :2])
    prefix[:, 2] = sample_logits(model.head_ntok(h[:, 1]).float(), temperature, rng)
    q_plan = dec_qtot(prefix[:, 1].cpu().numpy()); n_plan = dec_ntok(prefix[:, 2].cpu().numpy())
    caches = [dict() for _ in model.blocks]
    h = model.trunk(model.embed_prefix(prefix), None, caches)
    ctx = h[:, -1]
    keys = torch.zeros(B, 0, model.d, device=device)
    key_t = torch.zeros(B, 0, dtype=torch.long, device=device)
    toks = [[] for _ in range(B)]
    alive = torch.ones(B, dtype=torch.bool, device=device)
    q_sum = np.zeros(B); last_t = np.zeros(B, np.int64)
    stop = [None] * B

    col_step = torch.full((N_PREFIX,), -1, dtype=torch.long, device=device)
    col_lm = torch.zeros(B, N_PREFIX, dtype=torch.bool, device=device)
    prev = None
    bud = None
    t_head = torch.zeros(B, dtype=torch.long, device=device)

    def fixed_pos(tok):
        ptr = tok[:, ix["ptr"]].cpu().numpy()
        vals = np.zeros((B, 3), np.int64); mask = np.zeros(B, bool)
        for b in range(B):
            p = int(ptr[b])
            if p > 0 and p <= len(toks[b]):
                parent = toks[b][-p]
                anc = vocab.anchor(parent, int(tok[b, ix["s"]]))
                d = np.array([vocab.dec_d(int(tok[b, ix[k]])) for k in ("dx", "dy", "dz")]).reshape(3)
                vals[b] = np.clip(anc + d, 0, vocab.g - 1); mask[b] = True
        return vals, mask

    pos_cache = {}

    def fx(name, col):
        def f(tok):
            if name == "ix":
                pos_cache["v"], pos_cache["m"] = fixed_pos(tok)
            v = torch.from_numpy(pos_cache["v"][:, col]).to(device); m = torch.from_numpy(pos_cache["m"]).to(device)
            return v, m
        return f

    def na_when_no_ptr(tok):
        m = tok[:, ix["ptr"]] == 0
        return torch.zeros(B, dtype=torch.long, device=device), m

    fixed = {"ix": fx("ix", 0), "iy": fx("iy", 1), "iz": fx("iz", 2), "s": na_when_no_ptr, "dx": na_when_no_ptr, "dy": na_when_no_ptr, "dz": na_when_no_ptr}
    for step in range(max_tokens):
        M = keys.shape[1]
        ptr_mask = torch.ones(B, M, dtype=torch.bool, device=device) if M else None
        force_cont = (prev[:, ix["lt"]] > 0) if (cont and prev is not None) else torch.zeros(B, dtype=torch.bool, device=device)
        t_rule = model.rule_t(prev) if (cont and prev is not None) else None
        if use_rules:
            tok = decode_one_rules(model, ctx, fixed, temperature, keys if M else None, ptr_mask, rng, sample_context(model, prev, t_head, B, device),
                                   suppress_eos=step < min_tokens, attr_temp=attr_temp, key_t=key_t if (M and causal_ptr) else None, birth_ptr=birth_ptr,
                                   s_end=vocab.s_end)
        else:
            cons = sampling_rules(model, vocab, t_head if time_order else torch.zeros_like(t_head), prev) if constraints else None
            tok = decode_one(model, ctx, fixed, temperature, keys if M else None, ptr_mask, rng, suppress_eos=step < min_tokens, attr_temp=attr_temp,
                             key_t=key_t if (M and causal_ptr) else None, force_cont=force_cont if cont else None, t_rule=t_rule, birth_ptr=birth_ptr, s_end=vocab.s_end,
                             cons=cons)
        kind = tok[:, ix["kind"]]
        done_now = alive & (kind == KIND_EOS)
        alive = alive & ((kind == KIND_STROKE) | (kind == KIND_CONT))
        tk = tok.cpu().numpy()
        for b in range(B):
            if done_now[b]:
                stop[b] = "eos"
            if alive[b]:
                toks[b].append(tk[b]); q_sum[b] += float(vocab.dec_log(tk[b, ix["lq"]], vocab.n_q, vocab.q_log_max)); last_t[b] = tk[b, ix["t"]]
                if (stop_at_plan == "q" and q_sum[b] >= q_plan[b]) or (stop_at_plan == "n" and len(toks[b]) >= n_plan[b]):
                    alive[b] = False; stop[b] = "plan"
        if not alive.any():
            break
        tok = torch.where(alive[:, None], tok, torch.zeros_like(tok)); tok[:, ix["kind"]] = torch.where(alive, torch.where(kind == KIND_CONT, kind, torch.ones_like(kind)), kind)
        prev = tok
        t_head = torch.where(alive & (tok[:, ix["kind"]] != KIND_CONT), tok[:, ix["t"]], t_head)
        if use_bud:
            rem_n = np.array([n_plan[b] - len(toks[b]) for b in range(B)]); rem_q = q_plan - q_sum
            bud = torch.from_numpy(budget_bins(rem_n, rem_q)).to(device)[:, None, :]
        lm_now = (tok[:, ix["lq"]] >= lm_lq) & alive if (use_lm and lm_lq >= 0) else torch.zeros(B, dtype=torch.bool, device=device)
        col_step = torch.cat([col_step, torch.tensor([step], device=device)]); col_lm = torch.cat([col_lm, lm_now[:, None]], 1)
        attn = None
        if use_lm:
            vis = (col_step < 0) | (col_step > step - window) | col_lm
            attn = vis[:, None, None, :]
        h = model.trunk(model.embed_strokes(tok[:, None, :], bud), attn, caches)
        ctx = h[:, -1]
        keys = torch.cat([keys, ctx[:, None, :]], 1)[:, -model.w_ptr:]
        key_t = torch.cat([key_t, tok[:, ix["t"]][:, None]], 1)[:, -model.w_ptr:]

        if caches[0]["k"].shape[2] > N_PREFIX + window + 256:
            keep = (col_step < 0) | (col_step > step - window) | col_lm.any(0)
            idx = torch.nonzero(keep).squeeze(1)
            for c in caches:
                c["k"] = c["k"][:, :, idx]; c["v"] = c["v"][:, :, idx]
            col_step = col_step[idx]; col_lm = col_lm[:, idx]
        if (step + 1) % prefix_refresh == 0:
            for b in range(B):
                prefix[b, 3] = enc_cum(min(q_sum[b] / max(q_plan[b], 1.0), 0.999)); prefix[b, 4] = enc_cum(min(len(toks[b]) / max(n_plan[b], 1.0), 0.999)); prefix[b, 5] = int(last_t[b])
            fresh = [dict() for _ in model.blocks]
            model.trunk(model.embed_prefix(prefix), None, fresh)
            for c, f in zip(caches, fresh):
                c["k"][:, :, :N_PREFIX] = f["k"]; c["v"][:, :, :N_PREFIX] = f["v"]
        if verbose and (step + 1) % 1000 == 0:
            log(f"  step {step + 1}: alive {int(alive.sum())}/{B}, photons so far {q_sum.mean():.3g} (plan {q_plan.mean():.3g})")
    stop = [r if r is not None else "max" for r in stop]
    return [np.array(t, dtype=np.int64).reshape(-1, len(vocab.names)) for t in toks], dict(q_plan=q_plan, n_plan=n_plan, stop=stop)


@torch.no_grad()
def check_rule_order(model):
    o, ai = list(model.order), model.ai
    if o.index(ai["t"]) > o.index(ai["iz"]) or ("lt" in ai and max(o.index(ai["t"]), o.index(ai["len"])) > o.index(ai["lt"])):
        raise ValueError("the sampling rules need t decoded before iz, and t and len before lt (model.attr_order); sample with --no-rules")


def sampling_rules(model, vocab, t_head, prev):
    return dict(t_cap=int(vocab.enc_t(12.0 - 1e-9)), t_head=t_head, lt_prev=None if prev is None or "lt" not in vocab.index else prev[:, vocab.index["lt"]],
                vocab=vocab)


def cone_iz_min(vocab, t_bin):
    from .common import C_CM_PER_NS, VOX_CM
    t_up = (t_bin.double() + 1.0) * vocab.t_max / vocab.n_t
    return torch.floor(vocab.g - 1 - C_CM_PER_NS * t_up / VOX_CM).long() - 1


def masked_or_cap(lg, allowed, fallback):
    none = ~allowed.any(1)
    if none.any():
        allowed = allowed.clone()
        allowed[none] = False
        allowed[none, fallback[none].clamp(0, lg.shape[1] - 1)] = True
    return lg.masked_fill(~allowed, float("-inf"))


def decode_one(model, ctx, fixed, temperature, keys, ptr_mask, rng, suppress_eos=False, attr_temp=None, key_t=None, force_cont=None, t_rule=None, birth_ptr=True, s_end=16,
               cons=None):
    B = ctx.shape[0]
    tok = torch.zeros(B, model.A, dtype=torch.long, device=ctx.device)
    x0 = model.h_proj(ctx)[:, None, :]
    order = getattr(model, "order", list(range(model.A)))
    is_cont = torch.zeros(B, dtype=torch.bool, device=ctx.device)
    for p, a in enumerate(order):
        xs = [x0] + [model.attr_in[b](tok[:, b])[:, None, :] for b in order[:p]]
        x = torch.cat(xs, 1) + model.slot.weight[None, : p + 1, :]
        for blk in model.attr_blocks:
            x = blk(x, None)
        y = model.attr_ln(x[:, p])
        name = model.names[a]
        temperature_a = (attr_temp or {}).get(name, temperature)
        if name == "kind":
            logits = model.heads[a](y).float()
            logits[:, 0] = float("-inf")
            if suppress_eos:
                logits[:, 2] = float("-inf")
            if force_cont is not None and logits.shape[1] >= 4:
                logits[:, 3] = torch.where(force_cont, logits[:, 3], torch.full_like(logits[:, 3], float("-inf")))
                logits[:, :3] = torch.where(force_cont[:, None], torch.full_like(logits[:, :3], float("-inf")), logits[:, :3])
            elif logits.shape[1] >= 4:
                logits[:, 3] = float("-inf")
            tok[:, a] = sample_logits(logits, temperature_a, rng)
            is_cont = tok[:, a] == KIND_CONT
            continue
        if name == "ptr":
            if keys is None:
                tok[:, a] = torch.where(is_cont, torch.ones_like(tok[:, a]), tok[:, a])
                continue
            q = model.ptr_q(y)
            s = torch.einsum("bd,bjd->bj", q, model.ptr_k(keys)) / (model.d ** 0.5)
            if key_t is not None and order.index(model.ai["t"]) < p:
                ptr_mask = ptr_mask & (key_t <= tok[:, model.ai["t"]][:, None])
            s = s.masked_fill(~ptr_mask, float("-inf"))
            none = (q @ model.ptr_none)[:, None] / (model.d ** 0.5)
            c = sample_logits(torch.cat([none, s], 1).float(), temperature_a, rng)
            M = keys.shape[1]
            val = torch.where(c > 0, M - (c - 1), torch.zeros_like(c))
            if not birth_ptr:
                val = torch.zeros_like(val)
            tok[:, a] = torch.where(is_cont, torch.ones_like(val), val)
        elif name == "s" and is_cont.any():
            tok[:, a] = sample_logits(model.heads[a](y).float(), temperature_a, rng)
            tok[:, a] = torch.where(is_cont, torch.full_like(tok[:, a], int(s_end)), tok[:, a])
        elif name == "t" and ((t_rule is not None and is_cont.any() and getattr(model, "t_window", 0) > 0) or cons is not None):
            lg = model.heads[a](y).float()
            if t_rule is not None and is_cont.any() and getattr(model, "t_window", 0) > 0:
                lg = torch.where(is_cont[:, None], model.cont_t_mask(lg, t_rule), lg)
            if cons is not None:
                b = torch.arange(lg.shape[1], device=lg.device)[None, :]
                ok = (b <= cons["t_cap"]) & torch.isfinite(lg)
                ok &= is_cont[:, None] | (b >= cons["t_head"][:, None])
                lg = masked_or_cap(lg, ok, torch.full_like(cons["t_head"], cons["t_cap"]))
            tok[:, a] = sample_logits(lg, temperature_a, rng)
        elif name == "iz" and cons is not None:
            lg = model.heads[a](y).float()
            b = torch.arange(lg.shape[1], device=lg.device)[None, :]
            ok = b >= cone_iz_min(cons["vocab"], tok[:, model.ai["t"]])[:, None]
            lg = masked_or_cap(lg, ok, torch.full_like(tok[:, a], lg.shape[1] - 1))
            tok[:, a] = sample_logits(lg, temperature_a, rng)
        elif name == "lt" and cons is not None:
            lg = model.heads[a](y).float()
            b = torch.arange(lg.shape[1], device=lg.device)[None, :]
            ok = torch.ones_like(lg, dtype=torch.bool)
            if cons["lt_prev"] is not None:
                ok &= ~is_cont[:, None] | (b <= cons["lt_prev"][:, None])
            if model.t_window > 0:
                nxt = tok[:, model.ai["t"]] + model.t_rule[tok[:, model.ai["len"]]] - model.t_win_lo
                ok &= ~(nxt > cons["t_cap"])[:, None] | (b == 0)
            lg = masked_or_cap(lg, ok, torch.zeros_like(tok[:, a]))
            tok[:, a] = sample_logits(lg, temperature_a, rng)
        else:
            tok[:, a] = sample_logits(model.heads[a](y).float(), temperature_a, rng)
        if name in fixed:
            v, m = fixed[name](tok)
            tok[:, a] = torch.where(m, v, tok[:, a])
    return tok


def decode_one_rules(model, ctx, fixed, temperature, keys, ptr_mask, rng, rctx, suppress_eos=False, attr_temp=None, key_t=None, birth_ptr=True, s_end=16,
                     forced=None):
    from .rules import constrain
    R = model.rules
    B = ctx.shape[0]
    tok = torch.zeros(B, model.A, dtype=torch.long, device=ctx.device)
    x0 = model.h_proj(ctx)[:, None, :]
    order = model.order
    is_cont = torch.zeros(B, dtype=torch.bool, device=ctx.device)
    for p, a in enumerate(order):
        xs = [x0] + [model.attr_in[b](tok[:, b])[:, None, :] for b in order[:p]]
        x = torch.cat(xs, 1) + model.slot.weight[None, : p + 1, :]
        for blk in model.attr_blocks:
            x = blk(x, None)
        y = model.attr_ln(x[:, p])
        name = model.names[a]
        ta = (attr_temp or {}).get(name, temperature)
        if name == "ptr" and forced is not None:
            tok[:, a] = forced[:, a]
            continue
        if name == "ptr":
            val = torch.zeros(B, dtype=torch.long, device=ctx.device)
            if keys is not None and birth_ptr:
                q = model.ptr_q(y)
                sc = torch.einsum("bd,bjd->bj", q, model.ptr_k(keys)) / (model.d ** 0.5)
                pm = ptr_mask
                if key_t is not None and order.index(model.ai["t"]) < p:
                    pm = pm & (key_t <= tok[:, model.ai["t"]][:, None])
                sc = sc.masked_fill(~pm, float("-inf"))
                none = (q @ model.ptr_none)[:, None] / (model.d ** 0.5)
                c = sample_logits(torch.cat([none, sc], 1).float(), ta, rng)
                val = torch.where(c > 0, keys.shape[1] - (c - 1), torch.zeros_like(c))
            tok[:, a] = torch.where(is_cont, torch.ones_like(val), val)
            continue
        lg = model.heads[a](y).float()
        if name == "kind" and suppress_eos:
            lg[:, KIND_EOS] = float("-inf")
        lg = constrain(R, model, name, lg, tok, rctx)
        if getattr(model, "strict", None) is not None and forced is None and name in ("oz", "len"):
            lg = model.strict.apply(name, lg, tok, model.ai)
        if forced is not None:
            tok[:, a] = forced[:, a]
            if name == "kind":
                is_cont = tok[:, a] == KIND_CONT
            continue
        val = sample_logits(lg, ta, rng)
        if name == "s":
            val = torch.where(is_cont, torch.full_like(val, int(s_end)), val)
        tok[:, a] = val
        if name == "kind":
            is_cont = val == KIND_CONT
        if name in fixed:
            v, m = fixed[name](tok)
            tok[:, a] = torch.where(m, v, tok[:, a])
    return tok
