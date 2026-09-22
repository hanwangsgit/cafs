# Checkpoint attribution and license scope

The repository's MIT license applies to original CAFS source code. It does not
replace upstream checkpoint terms. Distribute this notice and `Apache-2.0.txt`
alongside the face checkpoints. The license text is stored under `LICENSES/`
in the source repository.

| File | Origin and modification | License status |
|---|---|---|
| `face_teacher.pt` | State-dictionary export of the `google/ddpm-celebahq-256` prior used in the experiments | Upstream model card declares Apache-2.0; retain that license and attribution |
| `face_cm.pt` | CAFS consistency-distilled student initialized from the face prior; weights were modified through training | Retain the upstream Apache-2.0 terms and this modification notice |
| `fmri.ckpt` | AdaSense pretrained single-coil knee diffusion prior | Upstream checkpoint terms; official source linked below |
| `mri_cm.pt` | CAFS consistency-distilled student initialized from `fmri.ckpt` | Teacher-derived weights; see the upstream-license note below |

Face prior: [google/ddpm-celebahq-256 model card](https://huggingface.co/google/ddpm-celebahq-256).
The model card credits Jonathan Ho, Ajay Jain, and Pieter Abbeel for
*Denoising Diffusion Probabilistic Models* (2020). Upstream model-card license
metadata was inspected on 2026-09-21. No separate upstream NOTICE file was
listed in the inspected model repository. This file records CAFS attribution
and modifications; it is not a substitute for any upstream notice.

MRI prior: Noam Elata, Tomer Michaeli, and Michael Elad,
*Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling* (2024).
[Official repository](https://github.com/noamelata/AdaSense) and
[teacher download linked by its README](https://drive.google.com/file/d/1Vzu0ixfV2CDnEGlSQjmlCOuw2gS10Ync/view).
The inspected AdaSense README provides the MRI teacher without an explicit
checkpoint redistribution license. Permission for redistribution of that file
and teacher-derived student weights has not been established. The source-code
MIT license does not grant rights to those weights.

No dataset files are included. Obtain datasets under their providers' terms.
