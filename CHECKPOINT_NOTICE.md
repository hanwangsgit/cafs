# Checkpoint attribution and license scope

The repository's MIT license applies to original CAFS source code. It does not
replace upstream checkpoint terms. Distribute this notice and `Apache-2.0.txt`
alongside the face checkpoints. The license text is stored under `LICENSES/`
in the source repository.

| File | Origin and modification | License status |
|---|---|---|
| `face_teacher.pt` | State-dictionary export of the `google/ddpm-celebahq-256` prior used in the experiments | Upstream model card declares Apache-2.0; retain that license and attribution |
| `face_cm.pt` | CAFS consistency-distilled student initialized from the face prior; weights were modified through training | Retain the upstream Apache-2.0 terms and this modification notice |
| `fmri.ckpt` | AdaSense's pretrained single-coil knee diffusion prior | Redistribution permission not established; obtain the teacher from its official source |
| `mri_cm.pt` | CAFS consistency-distilled student initialized as a copy of `fmri.ckpt`; weights were modified through training | Permission to redistribute these teacher-derived weights remains unresolved |

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
The inspected README provides the checkpoint but does not state a redistribution
license. The MIT license of the DDIM network implementation does not establish
a license for trained MRI weights.

Keep the mixed checkpoint folder restricted pending MRI permission. Face files
can be staged separately with their Apache license and this notice. No dataset
files are included, and no license is granted here for upstream datasets.
