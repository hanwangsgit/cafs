# Exact model inputs

The runner checks checkpoint **file** SHA-256 before loading. It never substitutes
an undistilled DDPM for a missing CM. Put weights in the ignored `checkpoints/`
directory or pass another local path. These files are not in Git.

| Input | SHA-256 |
|---|---|
| Face CM (`ema_step10000.pt`) | `2fcab89eb72c87d8063c72881b824c13f86296d7a89a2a1f3d7d4b94e089dd98` |
| Face teacher state dictionary | `7feaf1992a34be4b17d41e6d90185876e8fddc58bdd2409c619a0b45b79890e9` |
| MRI CM | `e295911e6a19c5ad93ee5e8380e7e8002b67648356e3569ee2de108e2f8ec9e0` |
| MRI teacher (`fmri.ckpt`) | `eec7efcb0dcad569b819c9f4d2cf311f11f7f8e5be9af50c6d002a0f2c55e1f4` |

**CM distribution is unresolved.** Obtain these exact trained student files from
the maintainer. No public URL is verified. Re-running distillation is not claimed
to recreate these files: checkpoint-specific training identities and complete
training provenance were not recorded. Training scripts with guessed settings
are intentionally not offered as exact reproduction.

For the face architecture, obtain a local Diffusers pipeline snapshot of
[`google/ddpm-celebahq-256`](https://huggingface.co/google/ddpm-celebahq-256).
The recorded `architecture_revision=c6a0e54d1d23` is a manifest-hash prefix, **not a
verified Hugging Face commit**. Its `model_index.json` must hash to
`c6a0e54d1d235280bce7e40abae49b73c4de0011ddbf09d817ce39e848cc2fa7`.
The original run then loaded explicit teacher/student state dictionaries into
the same architecture. Obtain the bound teacher dictionary from the maintainer;
re-serializing equivalent tensors can change a file hash.

The MRI teacher is linked from the [AdaSense pretrained-model instructions](https://github.com/noamelata/AdaSense#pretrained-models).
Verify the downloaded file against the table above. The student is a state dict
for `build_mri_unet()` (keys under `model.`); the teacher uses the underlying
network keys, optionally prefixed with `module.` or nested under `state_dict`.

```sh
shasum -a 256 checkpoints/face_cm.pt checkpoints/face_teacher.pt
shasum -a 256 checkpoints/mri_cm.pt checkpoints/fmri.ckpt
```

The local face CM file matches the expected hash. A local Hugging Face cache
snapshot named `cd5c944777ea2668051904ead6cc120739b86c4d` also has the expected
architecture-manifest hash. Its pipeline loads offline, and the face CM state
dictionary loads strictly with all keys matching under the tested environment.
This verifies loading, not a trained-model reconstruction or the full original
architecture-directory identity.

The cached `diffusion_pytorch_model.bin` hashes to
`efff89712093ad060ce99d9b461bbe542b49d8dd4ce30f23fe5761dca292361d`.
It is not the bound face teacher file listed above; tensor equivalence has not
been verified. Do not replace the required teacher hash with this candidate.
The exact face teacher and both MRI checkpoints were recovered from the cluster
and independently verified locally against the hashes above. All four files are
now staged under the ignored `checkpoints/` directory using the names in the
README, with `checkpoints/SHA256SUMS`. The downloaded archives and backup copies
remain under ignored `results/recovery/`. These local files are not Git assets
and no public checkpoint download links have yet been established.

CPU smoke runs with the verified weights completed five K=1 CAFS sensing rounds
and the paper's final reconstructor for each dataset. These validate execution;
CPU RNGs and the local package versions do not reproduce the original CUDA run.

Keep upstream model and dataset terms when obtaining or redistributing assets.
