# Reproduction and local verification

## Source and scope

The manuscript is *Fast Adaptive Fourier Sensing with Consistency Models*,
ICASSP 2027 draft, inspected on 2026-09-21. Its retained records point to research
commit `5dd9e6b2ee288f6f9bc8a60078f6d47877808a08`. Extraction reads that commit;
it does not use the later research branch defaults or alter the research tree.

`reproduction/source_map.json` maps 71 extracted definitions plus the MRI network
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
- All selector/final seeds, initial actions, round costs, and reconstructed mask
  hashes are checked across 180 original blocks (1,440 policy histories). All
  180 MRI uniform/VD histories match the extracted static selectors exactly.
  Six compact cases remain in `reproduction/reference_cases.json` for tests
  without the downloaded archives.
- All 42 quality rows and 14 efficiency rows reproduce manuscript rounding from
  the original records. `reproduction/verified_tables.json` stores the recomputed
  statistics; `verify_results.py` repeats the checks on extracted archives.
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

Run the retained checks with `python -m unittest discover -s tests -v`.
Local CPU environment: Python 3.10, torch 2.11.0, torchvision 0.26.0, NumPy 2.2.6,
Diffusers 0.37.1, datasets 4.8.4, h5py 3.16.0, Pillow 12.2.0. These are the pins in
`requirements.txt`, not a recovered historical GPU lockfile. The source records
report Python 3.10.20, torch 2.5.1+cu121 and CUDA 12.1; remaining historical package
versions are unavailable. No dependency installation was tested in a fresh
networked environment.

## Limits that remain

The recovered archives contain 90 complete face blocks and 90 complete MRI
blocks, eight sensing policies and two final reconstructors per block. The
original MRI launch package and both data manifests are available locally under
ignored `results/recovery/`. The release keeps numerical summaries, identifiers,
and hashes; it does not include raw archives or cluster launch infrastructure.

All 30 face source-file hashes and all 30 MRI source-file/tensor hashes have been
restored from the original bindings. The loaders enforce these identities.
Face exports also carry the pinned dataset revision and their per-file manifest.

Release validation uses the archived GPU records, source comparisons, checkpoint
hashes, and completed CPU checks described above. No further experiment runs are
planned for this release. The extracted package has not been rerun on CUDA, so
no new GPU masks, reconstruction metrics, memory measurements, or timings are
claimed. CPU/device/version equivalence must not be inferred from the smoke
checks. The full original face architecture-directory identity remains
unverified beyond the recorded manifest hash and successful model loading.
The maintainer-provided Google Drive folder is linked in `CHECKPOINTS.md`.
All four local Drive copies match the recorded hashes; anonymous cloud access
and downloads have not yet been independently verified. Face checkpoint license
notices are prepared. MRI teacher and distilled-student redistribution permission
remains unresolved; the mixed folder is not cleared for public distribution.

Full acquisition timing excludes setup and final reconstruction and subtracts
CM diagnostics. Paper timing uses medians on A100-SXM4 at 10% (9 faces, 24 MRI
volumes); memory uses the maximum across all 30 cases, three budgets and
recorded GPUs. `summarize.py` reports actual counts for new runs.

To verify the recovered records locally:

```sh
python verify_results.py results/recovery/face results/recovery/mri
```

## Local release state

The GitHub remote is `hanwangsgit/cafs`, and the initial release candidate has
been pushed. Original CAFS code is licensed under MIT in `LICENSE`; third-party
terms and remaining provenance questions are documented in `THIRD_PARTY.md`.
The manuscript still has placeholder authors, so this extraction does not invent
a final paper author list or bibliographic acceptance claim.

The repository uses a fresh `main` history containing only the release files.
The research repository history is not imported.

Remaining release tasks concern checkpoint access and third-party redistribution
terms. A new experiment run is not a release requirement; the verification scope
and limitations above remain part of the release documentation.
