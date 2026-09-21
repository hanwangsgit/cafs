# CAFS

Consistency Model-based Adaptive Fourier Sensing, accompanying
**Fast Adaptive Fourier Sensing with Consistency Models** (ICASSP 2027 manuscript).
Maintained by [hanwangsgit](https://github.com/hanwangsgit).

CAFS alternates a corrected few-step consistency-model estimate with Fourier
energy ranking. This extraction includes CelebA-HQ and single-coil fastMRI,
matched AdaSense/ADS selectors, and the paper's fixed sampling baselines.

**Release candidate:** the original source and all 180 result records have been
recovered. Their aggregates reproduce every published table entry at its stated
precision, and trained-model CPU smoke runs complete for both datasets.
Release validation uses these archived results and completed CPU checks.
The extracted package has not been rerun on CUDA; see [verification details](REPRODUCIBILITY.md).
Exact CM checkpoints must be supplied separately; public download links are not
yet available. Original CAFS code is [MIT licensed](LICENSE); third-party terms
and remaining provenance questions are documented in [THIRD_PARTY.md](THIRD_PARTY.md).

## Setup

Use Python 3.10 and a PyTorch-supported device. The full paper grid needs CUDA;
matched ADS used up to about 34 GiB allocated GPU memory.

```sh
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

## Data and checkpoints

Follow [CHECKPOINTS.md](CHECKPOINTS.md) for exact hashes and model preparation.
Obtain fastMRI single-coil knee validation HDF5 files through
[fastMRI](https://fastmri.med.nyu.edu/); place them under `data/singlecoil_val/`.
The pinned volume/slice identifiers are in `reproduction/mri_samples.json`.
Export the pinned CelebA-HQ images with:

```sh
python prepare_data.py --output data/celeba
```

## Reproduce

These commands run 30 cases × 3 budgets × 8 methods (including the one-shot
ablation). Final reconstruction is 20-step DDRM for faces and 10-step CM for MRI.

```sh
python run.py --domain face --data-root data/celeba \
  --architecture checkpoints/ddpm-celebahq-256 \
  --student checkpoints/face_cm.pt --teacher checkpoints/face_teacher.pt \
  --output results/face
python run.py --domain mri --data-root data/singlecoil_val \
  --student checkpoints/mri_cm.pt --teacher checkpoints/fmri.ckpt \
  --output results/mri
python summarize.py results/face > results/face-quality.csv
python summarize.py results/mri > results/mri-quality.csv
python summarize.py results/mri --efficiency > results/mri-efficiency.csv
```

For one CAFS example, append `--index 0 --budget 0.1 --policies cm_spectral_repeat_k5`.
Use a new output directory for each invocation. Records include masks, reconstructions,
metrics, per-round seeds, selection histories, NFEs, timing, and peak allocated memory.
`reproduction/paper_metrics.csv` contains the manuscript's rounded table values.
`reproduction/verified_tables.json` contains independently recomputed archive means
and efficiency statistics. Recheck extracted original archives with
`python verify_results.py results/recovery/face results/recovery/mri`.
The favorable figure cases are face index **8** and MRI index **16**, both at 10%.

Implementation: `cafs/reconstruction.py` predicts and corrects;
`cafs/policies.py` ranks complete Fourier groups;
`cafs/experiment.py` binds the paper's settings and RNG streams.
No research repository, cluster scheduler, or upstream baseline checkout is needed.
