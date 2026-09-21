# Attribution and release terms

- **MRI network (`cafs/mri_prior.py`)**: the DDIM architecture by Jiaming Song,
  through [AdaSense](https://github.com/noamelata/AdaSense/tree/45f9f96600570f745b3b5f8c32dbecccef8f9af0).
  [DDIM](https://github.com/ermongroup/ddim/blob/main/models/diffusion.py) is MIT;
  its notice is retained in `LICENSES/DDIM-MIT.txt`. The research version adds
  rectangular MRI geometry, a wrapper, checkpoint loading, and gradient checkpointing.
- **AdaSense selection adapter**: based on Noam Elata, Tomer Michaeli, and Michael
  Elad, *Adaptive Compressed Sensing with Diffusion-Based Posterior Sampling*
  (2024). Settings trace to commit `45f9f96600570f745b3b5f8c32dbecccef8f9af0`,
  `configs/celeba_hq.yml` and `configs/fmri.yml`. This is the research project's
  PyTorch selection port, not the full upstream pipeline.
- **ADS selection adapter**: based on [Active Diffusion Subsampling](https://github.com/active-diffusion-subsampling/ads/tree/64f2889b192cee2280ec7bf1405752b81ec02bc1)
  (Nolan et al., 2025), including whole-line positive-kernel entropy scoring.
  Settings trace to commit `64f2889b192cee2280ec7bf1405752b81ec02bc1`,
  `configs/benchmark/fastmri/fastmri_ads_10k.yaml`. Face ADS is a cross-domain
  adaptation. The upstream TensorFlow/Keras implementation is not bundled.
- **Final DDRM**: the research project's orthogonal measurement
  prediction–projection–renoising implementation, based on
  [DDRM](https://github.com/bahjat-kawar/ddrm) (Kawar et al., 2022).
- **Face prior**: [google/ddpm-celebahq-256](https://huggingface.co/google/ddpm-celebahq-256),
  loaded through Hugging Face Diffusers. Model and dataset terms remain separate
  from this repository's source-code license.

A standalone license was not found on the inspected AdaSense repository page;
fetching its pinned `LICENSE` and the ADS pinned `LICENSE` did not succeed.
The DDIM MIT notice covers its upstream architecture, but is not a blanket
license for other authors' modifications. Confirm the provenance/redistribution
rights of any AdaSense-derived changes before publishing. Original CAFS code is
licensed under the [MIT License](LICENSE). Third-party material retains its own
copyright and license terms; the CAFS license does not relicense that material.
No datasets or checkpoint files are distributed here.
