"""Model construction."""
from __future__ import annotations


import numpy as np
import torch

from .data import N_CUM, N_NTOK, N_QTOT
from .model import ShowerLM


def device_for(allow_cpu):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if allow_cpu:
        return torch.device("cpu")
    raise SystemExit("no GPU (use --allow-cpu for a test run)")


def is_detail(cfg):
    return cfg["model"].get("type", "showerlm") == "detail"


def build_model(cfg, vocab, device):
    m = cfg["model"]
    cls, extra = ShowerLM, {}
    if is_detail(cfg):
        from .detail import DetailLM
        cls, extra = DetailLM, dict(k_max=int(m.get("k_max", 1024)))
        if extra["k_max"] > vocab.w_ptr:
            raise SystemExit(f"model.k_max {extra['k_max']} > w_ptr {vocab.w_ptr}: the pointer attribute holds k + 1")
    model = cls(vocab.sizes, vocab.names, n_t=vocab.n_t, n_qtot=N_QTOT, n_ntok=N_NTOK, n_cum=N_CUM, d=m["d"], n_layers=m["n_layers"],
                     n_heads=m["n_heads"], d_attr=m["d_attr"], n_attr_layers=m["n_attr_layers"], n_attr_heads=m.get("n_attr_heads", 4),
                     window=m["window"], w_ptr=vocab.w_ptr, dropout=m.get("dropout", 0.0), attr_order=m.get("attr_order"), causal_ptr=m.get("causal_ptr", False),
                     budget=bool(m.get("budget", False)), landmarks=bool(m.get("landmarks", False)), t_window=m.get("t_window", 0),
                     t_rule_table=(vocab.t_bins_of_len(np.arange(vocab.n_len)).tolist() if vocab.v3 else None), q_weight_pow=float(m.get("q_weight_pow", 0.0)),
                     t_rel=bool(m.get("t_rel", False)), **extra)
    model.cont_top_extra = int(m.get("cont_top_extra", 0))
    if m.get("rules"):
        from .rules import Rules, check_order
        model.rules = Rules(vocab, m["rules"], model, detail=is_detail(cfg))
        check_order(model)
        if not is_detail(cfg) and model.rules.on["time_order"] and not cfg.get("data", {}).get("whole_sequence", False):
            raise SystemExit("rules.time_order on a ShowerLM needs data.whole_sequence (the floor is counted from the first stroke)")
    if model.t_rel and not (model.rules is not None and model.rules.on["time_order"]):
        raise SystemExit("model.t_rel needs rules.time_order (the t head predicts the offset above that floor)")
    model.landmark_lq = int(cfg.get("data", {}).get("landmark_lq", -1)) if bool(m.get("landmarks", False)) else -1
    return model.to(device)


def seq_len_at(schedule, step):
    cur = schedule[0][1]
    for s, L in schedule:
        if step >= s:
            cur = L
    return int(cur)
