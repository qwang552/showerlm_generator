"""3D views of generated showers in metres."""
import numpy as np

GRID = 192
Q_EPS = 0.001
VIRIDIS = [(0.2803, 0.1657, 0.4765), (0.2412, 0.2965, 0.5397), (0.1872, 0.4147, 0.5565),
           (0.1448, 0.5191, 0.5566), (0.1201, 0.6222, 0.5349), (0.2140, 0.7221, 0.4696),
           (0.4310, 0.8085, 0.3465), (0.7099, 0.8688, 0.1693), (0.9932, 0.9062, 0.1439)]
LUT_SCALE = 0.6


def lut256():
    a = np.asarray(VIRIDIS)
    t = np.linspace(0, 1, len(a))
    x = np.linspace(0, 1, 256)
    return np.stack([np.interp(x, t, a[:, k]) for k in range(3)], 1) * LUT_SCALE


def load(path):
    with np.load(path) as d:
        return d["ijk"].astype(np.int64), d["q"].astype(np.float64)


def colour_range(qs):
    pool = np.concatenate([np.log(np.asarray(q, dtype=np.float64) + Q_EPS) for q in qs])
    lo, hi = np.quantile(pool, [0.002, 0.999])
    return float(lo), float(hi if hi > lo else lo + 1)


def colour_index(q, lo, hi):
    lq = np.log(np.asarray(q, dtype=np.float64) + Q_EPS)
    return np.clip(np.rint((lq - lo) / (hi - lo) * 255), 0, 255).astype(np.int64)


def render(ijk, ci, lut, yaw=-0.6, pitch=0.3, size=520, zoom=1.0, gain=1.8, ps=2, grid=GRID):
    W = H = int(size)
    h = grid / 2
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    sc = zoom * min(W, H) / (grid * 1.85)

    def proj(x, y, z):
        x, y, z = x - h, y - h, z - h
        x1, y1 = x * cy - y * sy, x * sy + y * cy
        return W / 2 + x1 * sc, H / 2 - (y1 * sp + z * cp) * sc

    ijk = np.asarray(ijk, dtype=np.float64).reshape(-1, 3)
    u, v = proj(ijk[:, 0], ijk[:, 1], ijk[:, 2])
    u, v = np.trunc(u).astype(np.int64), np.trunc(v).astype(np.int64)
    col = lut[np.asarray(ci, dtype=np.int64)]
    buf = np.zeros((3, H * W))
    for a in range(ps):
        for b in range(ps):
            uu, vv = u + a, v + b
            ok = (uu >= 0) & (vv >= 0) & (uu < W) & (vv < H)
            k = vv[ok] * W + uu[ok]
            for ch in range(3):
                buf[ch] += np.bincount(k, weights=col[ok, ch], minlength=H * W)
    img = np.stack([5 + 250 * (1 - np.exp(-gain * buf[0])), 7 + 248 * (1 - np.exp(-gain * buf[1])),
                    10 + 245 * (1 - np.exp(-gain * buf[2]))], -1).reshape(H, W, 3) / 255.0
    return img, proj


def draw_box(ax, proj, ranges=((-2, 2), (-2, 2), (46, 50)), grid=GRID):
    G = grid
    R = np.asarray(ranges, dtype=np.float64)
    edges = [[0, 0, 0, G, 0, 0], [0, G, 0, G, G, 0], [0, 0, G, G, 0, G], [0, G, G, G, G, G], [0, 0, 0, 0, G, 0],
             [G, 0, 0, G, G, 0], [0, 0, G, 0, G, G], [G, 0, G, G, G, G], [0, 0, 0, 0, 0, G], [G, 0, 0, G, 0, G],
             [0, G, 0, 0, G, G], [G, G, 0, G, G, G]]
    for e in edges:
        a, b = proj(*e[:3]), proj(*e[3:])
        ax.plot([a[0], b[0]], [a[1], b[1]], color=(0.63, 0.67, 0.74, 0.35), lw=0.7)
    o = proj(0, 0, 0)
    vox = (R[:, 1] - R[:, 0]) / G
    for k, (lab, p, cl) in enumerate((("x", (G * 1.1, 0, 0), "#e66767"), ("y", (0, G * 1.1, 0), "#3fbf8a"), ("z", (0, 0, G * 1.1), "#5b9cf0"))):
        q = proj(*p)
        ax.plot([o[0], q[0]], [o[1], q[1]], color=cl, lw=1.2)
        ax.text(q[0] + 4, q[1] - 4, f"{lab} [m]", color=cl, fontsize=8)
        for m in np.arange(np.ceil(R[k, 0]), np.floor(R[k, 1]) + 0.5, 1.0):
            pt = [0.0, 0.0, 0.0]
            pt[k] = (m - R[k, 0]) / vox[k]
            t = proj(*pt)
            dx, dy = {0: (0, 9), 1: (7, 5), 2: (-12, 0)}[k]
            ax.plot([t[0]], [t[1]], marker="o", ms=1.6, color=cl)
            ax.text(t[0] + dx, t[1] + dy, f"{m:g}", color=cl, fontsize=5.5, ha="center", va="center", alpha=0.85)


def gallery(paths, cols=4, size=520, lq=None, out=None):
    """One panel per shower file, `cols` per row; saves the figure when `out` is given."""
    import matplotlib.pyplot as plt
    showers = [load(p) for p in paths]
    lq = lq or colour_range([q for _, q in showers])
    lut = lut256()
    cols = max(1, min(cols, len(showers)))
    rows = -(-len(showers) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.3 * rows), squeeze=False)
    fig.patch.set_facecolor("#0f1216")
    for k, ax in enumerate(axes.ravel()):
        ax.set_facecolor("#05070a")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
        if k >= len(showers):
            ax.set_visible(False)
            continue
        ijk, q = showers[k]
        img, proj = render(ijk, colour_index(q, *lq), lut, size=size)
        ax.imshow(img, interpolation="nearest")
        draw_box(ax, proj)
        ax.set_xlim(0, size)
        ax.set_ylim(size, 0)
    fig.tight_layout()
    if out:
        fig.savefig(out, dpi=150, facecolor=fig.get_facecolor())
    return fig
