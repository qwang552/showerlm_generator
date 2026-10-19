"""Plan and budget encodings used by the models."""
from __future__ import annotations


import numpy as np


N_QTOT, N_NTOK, N_CUM = 64, 64, 33
QTOT_LOG = (6.0, 9.0)
NTOK_LOG = (2.0, 5.0)
N_BUD = 40
BUD_N_MAX, BUD_Q_MAX = 2.0e4, 1.0e9


def enc_lin(x, lo, hi, n):
    return int(np.clip(np.floor((x - lo) / (hi - lo) * n), 0, n - 1))


def dec_lin(b, lo, hi, n):
    return lo + (np.asarray(b, dtype=np.float64) + 0.5) * (hi - lo) / n


def enc_qtot(q):
    return enc_lin(np.log10(max(q, 1.0)), *QTOT_LOG, N_QTOT)


def dec_qtot(b):
    return 10.0 ** dec_lin(b, *QTOT_LOG, N_QTOT)


def enc_ntok(n):
    return enc_lin(np.log10(max(n, 1.0)), *NTOK_LOG, N_NTOK)


def dec_ntok(b):
    return 10.0 ** dec_lin(b, *NTOK_LOG, N_NTOK)


def enc_cum(f):
    return int(np.clip(np.floor(f * (N_CUM - 1)), 0, N_CUM - 1))


def enc_bud(x, xmax):
    x = np.maximum(np.asarray(x, dtype=np.float64), 0.0)
    return np.clip(np.floor(np.log1p(x) / np.log1p(xmax) * (N_BUD - 1) + 1e-9), 0, N_BUD - 1).astype(np.int64)


def budget_bins(rem_n, rem_q):
    return np.stack([enc_bud(rem_n, BUD_N_MAX), enc_bud(rem_q, BUD_Q_MAX)], -1)
