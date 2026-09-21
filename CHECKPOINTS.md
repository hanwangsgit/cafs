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

The local face CM file was checked against the expected hash during extraction.
The other checkpoints and the bound face architecture were not available for
local end-to-end verification. Keep upstream model and dataset terms when
obtaining or redistributing any assets.
