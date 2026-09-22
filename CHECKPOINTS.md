# Data and model setup

Run the commands from the repository root after installing the dependencies.
The paper uses four checkpoint files. Their SHA-256 hashes are checked before
model loading.

## Checkpoints

Download the files and `SHA256SUMS` from the
[checkpoint folder](https://drive.google.com/drive/folders/1q_Hop4FCYnTJLEMBuSYFL6_cPjVjTo3p)
and place them in `checkpoints/`:

| File | Used for | SHA-256 |
|---|---|---|
| `face_cm.pt` | Face CAFS estimates | `2fcab89eb72c87d8063c72881b824c13f86296d7a89a2a1f3d7d4b94e089dd98` |
| `face_teacher.pt` | Face DDRM, matched AdaSense/ADS | `7feaf1992a34be4b17d41e6d90185876e8fddc58bdd2409c619a0b45b79890e9` |
| `mri_cm.pt` | MRI CAFS estimates and final CM | `e295911e6a19c5ad93ee5e8380e7e8002b67648356e3569ee2de108e2f8ec9e0` |
| `fmri.ckpt` | MRI matched AdaSense/ADS | `eec7efcb0dcad569b819c9f4d2cf311f11f7f8e5be9af50c6d002a0f2c55e1f4` |

```sh
cd checkpoints
shasum -a 256 -c SHA256SUMS
cd ..
```

Linux users can use `sha256sum -c SHA256SUMS`. The MRI teacher is also available
from the [official AdaSense download](https://drive.google.com/file/d/1Vzu0ixfV2CDnEGlSQjmlCOuw2gS10Ync/view).
Use the exact checkpoint files: re-saving the same tensors can change the file
hash. Checkpoint origins and license terms are in [CHECKPOINT_NOTICE.md](CHECKPOINT_NOTICE.md).
The local Drive copies match these hashes; anonymous cloud downloads have not
been independently verified.

## Face architecture

Download the exact Hugging Face snapshot used in the completed local loading
checks. The runner loads its architecture and schedule, then replaces the UNet
weights with the explicit student or teacher checkpoint above.

```sh
hf download google/ddpm-celebahq-256 \
  model_index.json config.json scheduler_config.json diffusion_pytorch_model.bin \
  --revision cd5c944777ea2668051904ead6cc120739b86c4d \
  --local-dir checkpoints/ddpm-celebahq-256
```

The `hf` command comes with the `huggingface_hub` dependency; see the
[official download instructions](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-download).
The runner requires `model_index.json` SHA-256
`c6a0e54d1d235280bce7e40abae49b73c4de0011ddbf09d817ce39e848cc2fa7`.
MRI architecture and schedule are included in `cafs/mri_prior.py`.

## CelebA-HQ

```sh
python prepare_data.py --output data/celeba
```

This exports the 30 specified images from `PhilSad/celeba-hq-1.5k`, revision
`a18e1cc5a351bce68739722bad0efa2c842bbc10`, and checks each PNG against its
recorded source hash. Images are converted to RGB, resized to 256×256 using
bilinear interpolation with antialiasing, and mapped to [-1,1] by the loader.
Image identifiers are in `reproduction/face_samples.json`.

## fastMRI

Obtain the single-coil knee validation HDF5 files through
[fastMRI](https://fastmri.med.nyu.edu/) under its data-access terms. Place the
files listed in `reproduction/mri_samples.json` in `data/singlecoil_val/`.
The manifest fixes one slice per volume. The loader verifies both the source
file and the preprocessed tensor hashes; it does not select a different slice
or normalization when an input differs. Datasets are not distributed here.
