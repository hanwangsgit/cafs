# Reproduction and local verification

## Source and scope

The manuscript is *Fast Adaptive Fourier Sensing with Consistency Models*,
ICASSP 2027 draft, inspected on 2026-09-21. Its retained records point to research
commit `5dd9e6b2ee288f6f9bc8a60078f6d47877808a08`. Extraction reads that commit;
it does not use the later research branch defaults or alter the research tree.

`reproduction/source_map.json` maps 70 extracted definitions plus the MRI network
file to their sources and hashes. Numerical definitions retain their source ASTs;
imports are local to `cafs`. The new experiment driver removes cluster integration,
DIP status machinery, unrelated experiments, and the unused second final
reconstructor. These removals do not consume or change selection RNG streams.
The one-shot K=1 arm is retained because the manuscript describes it.

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

- AST equality for every extracted numerical definition; byte equality for the
  MRI network against the recorded source commit.
- CPU original-versus-extracted comparisons: corrected CM at K=1/5/10 in both
  domains, final DDRM and AdaSense posterior draws in both domains, and MRI
  metrics. All 11 comparisons were bit-exact using deterministic toy priors.
- Tests exercise correction, timesteps, grouped budgets, all six adaptive arms,
  baseline NFE accounting, manifest cardinality, and summary aggregation.
- All selector/final seeds, initial action IDs, and round costs match six unique
  saved manuscript cases. Uniform and VD MRI histories match exactly for five
  saved volumes. These are recorded in `reproduction/reference_cases.json`.
- The local face student file SHA-256 matches its recorded checkpoint hash.
- No datasets, trained weights, cluster configs, private notes, or source Git
  history are included. Numeric reference cases contain sampling histories and
  metrics only, without images or machine/user paths.

Run the retained checks with `python -m unittest discover -s tests -v`.
Local CPU environment: Python 3.10, torch 2.11.0, torchvision 0.26.0, NumPy 2.2.6,
Diffusers 0.37.1, datasets 4.8.4, h5py 3.16.0, Pillow 12.2.0. These are the pins in
`requirements.txt`, not a recovered historical GPU lockfile. The source records
report Python 3.10.20, torch 2.5.1+cu121 and CUDA 12.1; remaining historical package
versions are unavailable. No dependency installation was tested in a fresh
networked environment.

## Limits that remain

The local result directories for jobs 2349848 (face) and 2514684 (MRI) contain
empty directory trees. The custom MRI launch package/bindings are also absent.
The MRI wrapper is reconstructed from the recorded config, saved histories and
shared source routines; full original-run equivalence is **not verified**.
The complete 180-block aggregate cannot be recomputed from the six saved cases.
`paper_metrics.csv` transcribes the manuscript tables; it is not a new experiment.

All 30 face IDs and all 30 MRI volume/slice pairs are recovered from manuscript
metadata. Original source-file hashes survive for only **2/30 faces** and
**5/30 MRI volumes**. Available hashes are enforced. Face exports also carry a
pinned dataset revision and a per-file manifest. Other original sample bytes
cannot be checked against the missing manifests; runtime records explicitly
report `paper_data_hash_verified` and the observed data hash. The saved MRI
`target_sha256` values are retained as provenance, not used as an unverified
cross-platform tensor-serialization check.

There is no local CUDA device, complete dataset, MRI checkpoint pair, or bound
face teacher/architecture for a full trained-model rerun. CUDA masks, PSNR/SSIM,
peak memory and elapsed time remain unverified. CPU/device/version equivalence
must not be inferred from the toy checks. Full acquisition timing excludes setup
and final reconstruction and subtracts CM diagnostics. Paper timing uses medians
on A100-SXM4 at 10% (9 faces, 24 MRI volumes); memory uses the maximum across all
30 cases, three budgets and recorded GPUs. `summarize.py` reports the actual
sample counts so an incomplete grid is visible.

The face CM training/evaluation overlap is unresolved. MRI uses a previously
accessed evaluation pool; checkpoint-specific training exclusions are not
independently verified. These data are not claimed to be an untouched test set.

## Local release state

The GitHub remote is `hanwangsgit/cafs`, and the initial release candidate has
been pushed. Original CAFS code is licensed under MIT in `LICENSE`; third-party
terms and remaining provenance questions are documented in `THIRD_PARTY.md`.
The manuscript still has placeholder authors, so this extraction does not invent
a final paper author list or bibliographic acceptance claim.

The repository uses a fresh `main` history containing only the release files.
The research repository history is not imported.

Resolve the asset and provenance gaps appropriate to the release claim before
making the repository public.
