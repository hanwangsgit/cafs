# Reproduction and local verification

## Source and scope

The implementation and archived measurements accompany *Fast Adaptive Fourier
Sensing with Consistency Models*. The numerical routines originate from research
commit `5dd9e6b2ee288f6f9bc8a60078f6d47877808a08`.

`reproduction/source_map.json` records source-file hashes and numerical-code
fingerprints. Documentation edits are tracked separately from executable code.
The standalone runner uses one common final reconstructor per dataset and retains
the one-shot K=1 ablation alongside the three repeated CAFS variants.

## Frozen settings

| Setting | CelebA-HQ | fastMRI |
|---|---|---|
| Cases | 30 RGB images, 256×256 | 30 single-coil knee volumes, one 640×368 slice each |
| Budget request | 5%, 10%, 25% | 5%, 10%, 25% |
| Actual real measurement count | 9,828 / 19,659 / 49,152 | 23,040 / 47,360 / 117,760 |
| MRI phase lines | — | 18 / 37 / 92 out of 368 |
| Initial acquisition | approximately 20% of final budget; DC mandatory | approximately 20% of final lines; at least 4; lines 366,367,0,1 mandatory |
| Additional rounds | 5, exact group-cost partition | 5, exact line-cost partition |
| CM K | 1, 5, 10 | 1, 5, 10 |
| CM schedule | `linspace(800,50,K+1).long()[:-1]` | same |
| Correction | zeta=1, clipping before/after correction to [-1,1] | zeta=1, no clipping |
| Final reconstruction | one 20-step DDRM sample, start 999 | one corrected 10-step CM sample |
| Metrics | RGB PSNR (range 2), local Gaussian SSIM | central 320×320 magnitude PSNR/SSIM/NMSE |
| Measurement noise | none | none |

The CM's last **evaluated** timestep is 800 for K=1, 200 for K=5, and 125 for
K=10. The endpoint 50 is a schedule boundary; changing this changes the experiment.
Face DDRM clips its prediction **before** projection, with no added final clamp.
MRI preprocessing uses NumPy inverse FFT (`norm='backward'`), two ifftshifts,
then division by `7.072103529760345e-07` and `0.3040714`. Do not replace this with
an orthonormal preprocessing FFT; sensing itself uses orthonormal FFTs.

AdaSense uses 8 samples × 25 source-rule DDRM timesteps per round, with no x0
clipping, for 1,000 sample NFEs. Its selection sampler is different from the
20-step final face sampler. ADS uses 16 jointly differentiated particles,
sigma=50, guidance=0.85, positive entropy, whole-action distance aggregation,
and the truncated 10,000-step schedule. Events are at 50, 663, 1275, 1888, 2500;
the inclusive prefix takes 2,501 forward and backward passes, or 80,032 sample
NFEs. Its suffix is discarded. MRI enables gradient checkpointing; ADS backward
nondeterminism can change masks across repeated GPU runs.

All blocks use protocol seed **0**, with SHA-256-derived 63-bit semantic streams
in `cafs/rng.py`; `cafs/experiment.py:seeds` specifies the domain-dependent keys.
Fixed face rankings consume the first selector stream. MRI fixed rankings use
a shared CPU float64 Gumbel stream. Initial-mask Gumbel draws use CPU float32,
as in the original. Do not unify these dtypes or move their RNGs between devices.

## Verification performed

- Executable-AST equality for every extracted numerical definition and the
  MRI network against the recorded source, excluding documentation strings.
- CPU original-versus-extracted comparisons: corrected CM at K=1/5/10 in both
  domains, final DDRM and AdaSense posterior draws in both domains, and MRI
  metrics. All 11 comparisons were bit-exact using deterministic toy priors.
- Tests exercise correction, timesteps, grouped budgets, all six adaptive arms,
  baseline NFE accounting, manifest cardinality, and summary aggregation.
- All selector/final seeds, initial actions, round costs, and reconstructed mask
  hashes are checked across 180 original blocks (1,440 policy histories). All
  180 MRI uniform/VD histories match the extracted static selectors exactly.
  Six compact cases remain in `reproduction/reference_cases.json` for tests
  without the downloaded archives.
- All 42 quality rows and 14 efficiency rows reproduce manuscript rounding from
  the original records. `reproduction/verified_tables.json` stores the recomputed
  statistics; `verify_results.py` repeats table recomputation using the bundled measurements
  and supports the additional audit when original archives are supplied.
- The original MRI runner, protocol, and eligibility file match their bound
  package hash. Every MRI record agrees with the recovered binding and protocol.
  The standalone wrapper has been reviewed against that recovered runner.
- All four checkpoint files match their recorded SHA-256 values locally. The
  face architecture manifest matches, and both domains' model pairs load.
- Trained-model CPU smoke runs completed five K=1 sensing rounds plus 20-step
  face DDRM or 10-step MRI CM at 10% sampling. MRI used its archived original
  target tensor; the face smoke used the paper's exported ground-truth PNG,
  not the bound original source PNG. These are execution checks, not GPU metric
  reproductions or full raw-data-loader tests.
- No datasets, trained weights, cluster configs, private notes, or source Git
  history are included. Numeric reference cases contain sampling histories and
  metrics only, without images or machine/user paths.

## Environments

| Profile | PyTorch / torchvision | Purpose |
|---|---|---|
| Original recorded runs | 2.5.1+cu121 / not recorded | Source of the reported GPU measurements |
| `requirements-cuda.txt` | 2.5.1+cu121 / 0.20.1+cu121 | Linux GPU installation using the recorded PyTorch/CUDA versions |
| `requirements.txt` | 2.11.0 / 0.26.0 | Environment used for the completed CPU checks |

Both release profiles use the shared pins in `requirements-runtime.txt`.
Those additional pins come from the local validation environment; they are not
an archived lockfile for every original GPU package. The CUDA profile has not
been executed during this extraction. Its PyTorch/torchvision pairing follows
the [official PyTorch 2.5.1 installation instructions](https://pytorch.org/get-started/previous-versions/#v251).

## Included records

`reproduction/paper_runs.csv` contains one row per domain, case, budget, and
sensing policy: 180 blocks and 1,440 rows. Each row preserves the original final
reconstructor's quality metrics, sensing NFEs, elapsed time, allocated memory,
GPU model, and SHA-256 of its source record. The face rows use final DDRM; MRI
rows use final corrected CM. Paths, hostnames, datasets, and weights are omitted.

```sh
python3 verify_results.py > verified-results.json
```

The command uses the bundled measurements and Python's standard library to
check coverage, duplicates, means, timing medians, peak memory, and agreement
with the rounded published tables. This is table recomputation from recorded
measurements; it performs no model execution. `reproduction/verified_tables.json`
contains the corresponding archive-verification output.

For maintainers with the original unpacked archives, the same script supports
an additional audit of action histories, masks, bindings, and the original MRI
source package:

```sh
python verify_results.py FACE_ARCHIVE_DIRECTORY MRI_ARCHIVE_DIRECTORY
```

This optional mode requires PyTorch and original archive metadata. It is not
needed for the default public table-verification command.

## Verification scope

The completed verification consists of source comparisons, archived-record
checks, exact checkpoint hashes, and CPU execution checks. The extracted package
has not been rerun on CUDA. Existing GPU results are supplied as measurements;
new runs can vary with device, software versions, and ADS backward nondeterminism.

All 30 face source-file hashes and all 30 MRI source-file/tensor hashes are
included in the sample manifests and enforced by the loaders. The face pipeline
snapshot specified in `configs/face.json` was loaded locally with the verified
student and teacher files. Its model-index hash matches the recorded binding;
the complete original pipeline-directory contents were not independently hashed.

Checkpoint locations and setup commands are in [CHECKPOINTS.md](CHECKPOINTS.md).
All four local checkpoint files and their copies in the Drive sync folder match
the expected hashes. Anonymous cloud downloads have not been independently
verified. Checkpoint licensing information is consolidated in
[CHECKPOINT_NOTICE.md](CHECKPOINT_NOTICE.md).
