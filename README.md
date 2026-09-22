# CAFS

Consistency Model-based Adaptive Fourier Sensing, accompanying
**Fast Adaptive Fourier Sensing with Consistency Models**.
Maintained by [hanwangsgit](https://github.com/hanwangsgit).

CAFS alternates a corrected few-step consistency-model estimate with Fourier
energy ranking. This extraction includes CelebA-HQ and single-coil fastMRI,
matched AdaSense/ADS selectors, and the paper's fixed sampling baselines.

## Results

At 10% sampling, five-step CAFS reduces median sensing time relative to matched
AdaSense by **1.22× on CelebA-HQ** and **24.24× on fastMRI**.

**Table 1. Sensing efficiency over five acquisition rounds.**

| Method | NFEs | CelebA-HQ peak GPU (GiB) | CelebA-HQ time (s) | fastMRI peak GPU (GiB) | fastMRI time (s) |
|---|---:|---:|---:|---:|---:|
| Uniform | 0 | 0.864 | 1.041 | 0.506 | 0.027 |
| VD | 0 | 0.867 | 1.229 | 0.510 | 0.014 |
| CAFS K=1 | 5 | 1.242 | 9.557 | 1.131 | 0.177 |
| **CAFS K=5** | 25 | 1.244 | 9.819 | 1.135 | 0.823 |
| CAFS K=10 | 50 | 1.244 | 10.442 | 1.135 | 1.632 |
| AdaSense | 1,000 | 3.672 | 12.008 | 5.574 | 19.952 |
| ADS | 80,032 | 33.823 | 860.255 | 29.073 | 2,335.080 |

Times exclude setup and final reconstruction and are medians at 10% sampling on
an NVIDIA A100-SXM4 (80 GB), over 9 face images and 24 MRI volumes. Peak memory is
the maximum allocated memory over all 30 cases, three budgets, and recorded GPU
types. NFEs count sample-level forward and backward evaluations. AdaSense and ADS
are matched selection adapters; face ADS adapts the MRI settings.

**Figure 2. CelebA-HQ reconstruction at 10% sampling.**

![Ground truth, Uniform, VD, CAFS K=5, AdaSense, and ADS face reconstructions, with PSNR and SSIM labels](assets/figure2-celeba.png)

Image `celeba-hq:1257`, with a shared 20-step DDRM final reconstructor.
Labels show PSNR (dB) / SSIM.

**Figure 3. fastMRI reconstruction at 10% sampling.**

![Ground truth, Uniform, VD, CAFS K=5, AdaSense, and ADS knee reconstructions, with PSNR and SSIM labels](assets/figure3-fastmri.png)

Volume `file1002145.h5`, slice 19, with a shared 10-step corrected CM final
reconstructor. Images show the central 320×320 magnitude crop; labels show
PSNR (dB) / SSIM.

Both figures are selected favorable CAFS examples and use a common display scale
within each panel. Aggregate quality over all 30 cases per dataset at 5%, 10%,
and 25% sampling is available in [the quality tables](reproduction/paper_metrics.csv).

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

## Reproducibility and licenses

All 180 original result records are recovered, and their aggregates reproduce
the reported tables at the stated precision. Source comparisons, checkpoint
hashes, and completed CPU checks are documented in
[REPRODUCIBILITY.md](REPRODUCIBILITY.md). The extracted package has not been rerun
on CUDA.

Original CAFS code is [MIT licensed](LICENSE). Checkpoints are staged in the
[Google Drive folder](https://drive.google.com/drive/folders/1q_Hop4FCYnTJLEMBuSYFL6_cPjVjTo3p).
See [checkpoint terms](CHECKPOINT_NOTICE.md) and [third-party attribution](THIRD_PARTY.md)
for the upstream licenses and unresolved MRI-weight redistribution permission.
