"""Stroke vocabulary: encoding and decoding of the 20 attributes."""
from __future__ import annotations


import numpy as np


KIND_PAD, KIND_STROKE, KIND_EOS, KIND_CONT = 0, 1, 2, 3
D_REP = np.array([-30, -20, -14, -11, -9] + list(range(-8, 9)) + [9, 11, 14, 20, 30])
COH_EDGES = np.array([0.0, 0.5, 0.8, 0.9, 0.95, 0.99, 0.999, 1.0001])


class Vocab:

    def __init__(self, g=192, t_max=13.0, n_t=512, len_max=32.0, n_len=65, w_ptr=2048, n_th=180, n_ph=360, n_q=96, q_log_max=7.0,
                 n_c=32, c_log_max=4.0, n_s=16, n_off=8, v3=False, n_lt=24, lt_max=400.0):
        self.g, self.t_max, self.n_t, self.len_max, self.n_len, self.w_ptr = int(g), float(t_max), int(n_t), float(len_max), int(n_len), int(w_ptr)
        self.n_th, self.n_ph, self.n_q, self.q_log_max, self.n_c, self.c_log_max = int(n_th), int(n_ph), int(n_q), float(q_log_max), int(n_c), float(c_log_max)
        self.n_s, self.n_off = int(n_s), int(n_off)
        self.v3, self.n_lt, self.lt_max = bool(v3), int(n_lt), float(lt_max)
        self.attrs = [("kind", 4 if self.v3 else 3), ("ptr", self.w_ptr + 1), ("s", self.n_s + 1), ("dx", len(D_REP) + 1), ("dy", len(D_REP) + 1), ("dz", len(D_REP) + 1),
                      ("ix", self.g), ("iy", self.g), ("iz", self.g), ("ox", self.n_off), ("oy", self.n_off), ("oz", self.n_off),
                      ("t", self.n_t), ("th", self.n_th), ("ph", self.n_ph), ("len", self.n_len), ("lq", self.n_q), ("lc", self.n_c), ("coh", len(COH_EDGES) - 1)]
        if self.v3:
            self.attrs.append(("lt", self.n_lt))
        self.names = [a for a, _ in self.attrs]
        self.sizes = [s for _, s in self.attrs]
        self.index = {a: i for i, a in enumerate(self.names)}

    def to_dict(self):
        d = dict(g=self.g, t_max=self.t_max, n_t=self.n_t, len_max=self.len_max, n_len=self.n_len, w_ptr=self.w_ptr, n_th=self.n_th, n_ph=self.n_ph,
                 n_q=self.n_q, q_log_max=self.q_log_max, n_c=self.n_c, c_log_max=self.c_log_max, n_s=self.n_s, n_off=self.n_off)
        if self.v3:
            d.update(v3=True, n_lt=self.n_lt, lt_max=self.lt_max)
        return d

    @property
    def s_end(self):
        return self.n_s

    def enc_lt(self, L):
        L = np.asarray(L, dtype=np.float64)
        b = 1 + np.floor(np.log(np.maximum(L, 1e-9)) / np.log(self.lt_max) * (self.n_lt - 2))
        return np.where(L <= 0, 0, np.clip(b, 1, self.n_lt - 1)).astype(np.int64)

    def dec_lt(self, b):
        b = np.asarray(b, dtype=np.float64)
        return np.where(b <= 0, 0.0, np.exp((np.maximum(b, 1) - 1 + 0.5) / (self.n_lt - 2) * np.log(self.lt_max)))

    def t_bins_of_len(self, len_bin):
        from .common import C_CM_PER_NS, VOX_CM
        L = (self.dec_len(len_bin) + 1.0) * VOX_CM
        return np.rint(L / C_CM_PER_NS / (self.t_max / self.n_t)).astype(np.int64)

    @classmethod
    def from_dict(cls, d):
        return cls(**d)


    def enc_t(self, t):
        return np.clip(np.floor(np.asarray(t, dtype=np.float64) / self.t_max * self.n_t), 0, self.n_t - 1).astype(np.int64)

    def dec_t(self, b):
        return (np.asarray(b, dtype=np.float64) + 0.5) * self.t_max / self.n_t

    def enc_dir(self, u):
        u = np.asarray(u, dtype=np.float64).reshape(-1, 3)
        th = np.degrees(np.arccos(np.clip(u[:, 2], -1, 1)))
        ph = np.degrees(np.arctan2(u[:, 1], u[:, 0]))
        bth = np.clip(np.floor(th / 180.0 * self.n_th), 0, self.n_th - 1).astype(np.int64)
        bph = np.clip(np.floor((ph + 180.0) / 360.0 * self.n_ph), 0, self.n_ph - 1).astype(np.int64)
        return bth, bph

    def dec_dir(self, bth, bph):
        th = np.radians((np.atleast_1d(bth).astype(np.float64) + 0.5) * 180.0 / self.n_th)
        ph = np.radians((np.atleast_1d(bph).astype(np.float64) + 0.5) * 360.0 / self.n_ph - 180.0)
        return np.stack([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)], -1)

    def enc_len(self, length):
        return np.clip(np.rint(np.asarray(length, dtype=np.float64) / self.len_max * (self.n_len - 1)), 0, self.n_len - 1).astype(np.int64)

    def dec_len(self, b):
        return np.asarray(b, dtype=np.float64) * self.len_max / (self.n_len - 1)

    def enc_log(self, x, n, lmax):
        return np.clip(np.floor(np.log10(np.maximum(np.asarray(x, dtype=np.float64), 1.0)) / lmax * n), 0, n - 1).astype(np.int64)

    def dec_log(self, b, n, lmax):
        return 10.0 ** ((np.asarray(b, dtype=np.float64) + 0.5) * lmax / n)

    def enc_coh(self, coh):
        return np.clip(np.searchsorted(COH_EDGES, np.asarray(coh, dtype=np.float64), side="right") - 1, 0, len(COH_EDGES) - 2)

    def dec_coh(self, b):
        b = np.asarray(b, dtype=np.int64)
        return 0.5 * (COH_EDGES[b] + np.minimum(COH_EDGES[b + 1], 1.0))

    def enc_off(self, off):
        return np.clip(np.floor((np.asarray(off, dtype=np.float64) + 0.5) * self.n_off), 0, self.n_off - 1).astype(np.int64)

    def dec_off(self, b):
        return (np.asarray(b, dtype=np.float64) + 0.5) / self.n_off - 0.5

    def enc_d(self, d):
        return np.argmin(np.abs(np.asarray(d, dtype=np.float64).reshape(-1)[:, None] - D_REP[None, :]), axis=1) + 1

    def dec_d(self, b):
        return D_REP[np.maximum(np.asarray(b, dtype=np.int64) - 1, 0)]

    def enc_s(self, s):
        return np.clip(np.floor(np.asarray(s, dtype=np.float64) * self.n_s), 0, self.n_s - 1).astype(np.int64) + 1

    def dec_s(self, b):
        return (np.maximum(np.asarray(b, dtype=np.float64) - 1, 0) + 0.5) / self.n_s


    def head_point(self, tok):
        tok = np.asarray(tok).reshape(-1, len(self.names))
        ijk = tok[:, [self.index["ix"], self.index["iy"], self.index["iz"]]].astype(np.float64)
        off = np.stack([self.dec_off(tok[:, self.index[a]]) for a in ("ox", "oy", "oz")], 1)
        return ijk + 0.5 + off

    def anchor(self, tok_parent, s_bin):
        head = self.head_point(tok_parent)[0]
        u = self.dec_dir(tok_parent[self.index["th"]], tok_parent[self.index["ph"]])[0]
        length = float(self.dec_len(tok_parent[self.index["len"]]))
        s = float(self.dec_s(s_bin)) if length > 0 else 0.0
        return np.clip(np.floor(head + s * length * u).astype(np.int64), 0, self.g - 1)

    def decode(self, tok, use_pointer=True):
        tok = np.asarray(tok, dtype=np.int64).copy()
        tok = tok[is_stroke(tok[:, self.index["kind"]])]
        ix = self.index
        if use_pointer:
            for i in range(len(tok)):
                p = int(tok[i, ix["ptr"]])
                if p > 0 and i - p >= 0:
                    a = self.anchor(tok[i - p], tok[i, ix["s"]])
                    d = np.array([self.dec_d(tok[i, ix[k]]) for k in ("dx", "dy", "dz")]).reshape(3)
                    tok[i, [ix["ix"], ix["iy"], ix["iz"]]] = np.clip(a + d, 0, self.g - 1)
        ijk = tok[:, [ix["ix"], ix["iy"], ix["iz"]]]
        off = np.stack([self.dec_off(tok[:, ix[a]]) for a in ("ox", "oy", "oz")], 1)
        u = self.dec_dir(tok[:, ix["th"]], tok[:, ix["ph"]])
        return dict(ijk=ijk, off=off, u=u, len=self.dec_len(tok[:, ix["len"]]), t=self.dec_t(tok[:, ix["t"]]),
                    Q=self.dec_log(tok[:, ix["lq"]], self.n_q, self.q_log_max), C=self.dec_log(tok[:, ix["lc"]], self.n_c, self.c_log_max),
                    coh=self.dec_coh(tok[:, ix["coh"]]), tok=tok)


def is_stroke(kind):
    kind = np.asarray(kind)
    return (kind == KIND_STROKE) | (kind == KIND_CONT)
