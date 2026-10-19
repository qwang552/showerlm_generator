# ShowerLM generator

Generates in-ice Cherenkov light showers in the format of the CORSIKA 8 voxel cache: photons per voxel in a
4 m × 4 m × 4 m box (192³ voxels of 2.08 cm), light emitted within the first 12 ns. The trained models are in
`trained_model/`; no input data is needed.

## Usage

```bash
pip install -r requirements.txt
python generate.py --n 10 --out showers --seed 0
```

| option | default | meaning |
|---|---|---|
| `--n` | 10 | number of showers |
| `--out` | `showers` | output folder (must not exist) |
| `--seed` | 0 | random seed; the same seed and batch give the same showers |
| `--batch` | 10 | showers generated together |
| `--no-strict` | off | disable the light-front and end-time sampling rules |
| `--no-t-caps` | off | disable the caps on the time step between detail strokes |

A GPU is used when available, otherwise the CPU. Measured times per shower:

| hardware | batch | time per shower |
|---|---|---|
| GTX 1080 | 25 | about 1 min |
| GTX 1080 | 10 | 2–2.7 min |
| CPU, 16 dedicated threads (AMD EPYC 7543P) | 1 | about 3 min |
| CPU on a shared login node | 1 | 7–10 min |

To view the showers, open `visualize.ipynb`.

## Output

`showers/sample_XXXXX.npz`, one file per shower:

| key | shape | meaning |
|---|---|---|
| `ijk` | (N, 3) | voxel indices 0–191 in x, y, z; x, y ∈ [−2, 2] m, z ∈ [46, 50] m; the primary enters at (0, 0, 50 m) moving down |
| `q` | (N,) | photons in the voxel |
| `off` | (N, 3) | photon-weighted position inside the voxel, in voxel units (−0.5 to 0.5) |
| `m` | (N, 3) | photon-weighted direction of the emitting particles; its length measures how aligned they are |
| `t` | (N,) | photon-weighted time minus the time of a plane front moving down at the speed of light (ns) |
| `cnt` | (N,) | number of emission records |

The folder also holds `tokens_XXXXX.npz` (the generated strokes), `run.json`, `timing.csv` and `summary.json`.

## Model

A shower is described as a sequence of straight light strokes: pieces of particle tracks up to 67 cm long, or single
voxels. Each stroke has 20 discrete attributes (time, position, direction, length, photons and others).

1. The skeleton model generates the long tracks, one stroke at a time.
2. The detail model generates the remaining strokes in time order, conditioned on the skeleton.
3. The strokes are drawn into the voxel grid.

Sampling enforces the light cone, the 12 ns window, time ordering and the maximum photon yield per record.
The checkpoints contain the averaged (EMA) weights in float16.
