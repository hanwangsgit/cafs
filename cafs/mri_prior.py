"""MRI diffusion prior: AdaSense's fastMRI knee teacher, adapted to our stack.

The `Model` UNet below (through `forward`) is vendored verbatim from
https://github.com/noamelata/AdaSense (models/diffusion.py), the ermongroup
DDIM architecture. The checkpoint `fmri.ckpt` (Google Drive link in their
README) is an eps-prediction DDPM trained on fastMRI singlecoil knee,
2-channel complex images at 640x368, linear beta schedule 1e-4..0.02 x1000
-- the same discrete schedule family as our face teacher, which is what
makes CM distillation warm-start possible.

Our additions at the bottom of the file:
  FMRI_CONFIG            the fmri.yml model config as nested namespaces
  mri_alphas_cumprod()   the (1000,) alpha-bar schedule
  EpsWrapper             makes Model return `.sample` like a diffusers UNet
  load_mri_teacher()     checkpoint -> wrapped eval-mode teacher + schedule
"""

import math
from collections.abc import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(
            x, scale_factor=2.0, mode="nearest")
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_


class Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        ch, out_ch, ch_mult = config.model.ch, config.model.out_ch, tuple(config.model.ch_mult)
        num_res_blocks = config.model.num_res_blocks
        attn_resolutions = config.model.attn_resolutions
        dropout = config.model.dropout
        in_channels = config.model.in_channels
        resolution = config.data.image_size if not (isinstance(config.data.image_size, Iterable)) else config.data.image_size[-1]
        resamp_with_conv = config.model.resamp_with_conv
        num_timesteps = config.diffusion.num_diffusion_timesteps

        if config.model.type == 'bayesian':
            self.logvar = nn.Parameter(torch.zeros(num_timesteps))
        
        self.ch = ch
        self.temb_ch = self.ch*4
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels
        # Opt-in only. Off, `_run_block` dispatches the vendored call
        # unchanged, so every existing caller keeps its exact bytes.
        self.gradient_checkpointing = False

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                if i_block == self.num_res_blocks:
                    skip_in = ch*in_ch_mult[i_level]
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def _run_block(self, block, *args):
        """Call one res/attn block, recomputing it in backward if asked.

        Storing every activation of this net for a 16-particle batch at
        640x368 needs ~78 GiB, which does not fit an 80 GB A100; the level-0
        blocks alone hold most of it. Recomputation trades ~1/3 more compute
        for that memory and leaves the gradient mathematically identical, so
        ADS keeps its single joint particle-batch L2 norm.
        """
        if self.gradient_checkpointing and torch.is_grad_enabled():
            return checkpoint(block, *args, use_reentrant=False)
        return block(*args)

    def forward(self, x, t):
        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)

        # downsampling
        hs = [self.conv_in(x)]
        for i_level in range(self.num_resolutions):
            for i_block in range(self.num_res_blocks):
                h = self._run_block(
                    self.down[i_level].block[i_block], hs[-1], temb)
                if len(self.down[i_level].attn) > 0:
                    h = self._run_block(self.down[i_level].attn[i_block], h)
                hs.append(h)
            if i_level != self.num_resolutions-1:
                hs.append(self.down[i_level].downsample(hs[-1]))

        # middle
        h = hs[-1]
        h = self._run_block(self.mid.block_1, h, temb)
        h = self._run_block(self.mid.attn_1, h)
        h = self._run_block(self.mid.block_2, h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self._run_block(
                    self.up[i_level].block[i_block],
                    torch.cat([h, hs.pop()], dim=1), temb)
                if len(self.up[i_level].attn) > 0:
                    h = self._run_block(self.up[i_level].attn[i_block], h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


# ---------------------------------------------------------------------------
# Our additions: config, schedule, wrapper, loader
# ---------------------------------------------------------------------------

from types import SimpleNamespace


def _ns(**kw):
    return SimpleNamespace(**kw)


# configs/fmri.yml from the AdaSense repo, fields the Model actually reads.
FMRI_CONFIG = _ns(
    data=_ns(dataset="MRI", image_size=[640, 368], channels=2),
    model=_ns(type="simple", in_channels=2, out_ch=2, ch=64,
              ch_mult=[1, 1, 2, 4, 8], num_res_blocks=2,
              attn_resolutions=[23], dropout=0.1,
              var_type="fixedsmall", ema_rate=0.9999, ema=True,
              resamp_with_conv=True),
    diffusion=_ns(beta_schedule="linear", beta_start=0.0001, beta_end=0.02,
                  num_diffusion_timesteps=1000),
)

MRI_H, MRI_W, MRI_C = 640, 368, 2


def mri_alphas_cumprod(config=FMRI_CONFIG) -> torch.Tensor:
    """The teacher's discrete alpha-bar schedule (linear betas)."""
    d = config.diffusion
    betas = torch.linspace(d.beta_start, d.beta_end,
                           d.num_diffusion_timesteps, dtype=torch.float32)
    return torch.cumprod(1.0 - betas, dim=0)


class EpsOutput:
    __slots__ = ("sample",)

    def __init__(self, sample):
        self.sample = sample


class EpsWrapper(nn.Module):
    """Give the DDIM `Model` the diffusers calling convention.

    cgmap_reconstruction / adaptive_sensing / the distillation loop all call
    `unet(x, t).sample`; the vendored Model returns a raw eps tensor.
    """

    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def enable_gradient_checkpointing(self) -> None:
        """Recompute block activations in backward instead of storing them."""
        self.model.gradient_checkpointing = True

    def forward(self, x, t):
        if not torch.is_tensor(t):
            t = torch.tensor([t], device=x.device)
        t = t.expand(x.shape[0]) if t.dim() == 0 or t.shape[0] == 1 else t
        return EpsOutput(self.model(x, t.float()))


def build_mri_unet(config=FMRI_CONFIG) -> EpsWrapper:
    """Fresh (untrained) wrapped UNet with the fmri architecture."""
    return EpsWrapper(Model(config))


def load_mri_teacher(ckpt_path: str, device: str = "cuda"):
    """Load fmri.ckpt -> (wrapped eval-mode teacher, alphas_cumprod)."""
    wrapped = build_mri_unet()
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    # Checkpoints saved from nn.DataParallel carry a 'module.' prefix.
    state = { (k[len("module."):] if k.startswith("module.") else k): v
              for k, v in state.items() }
    wrapped.model.load_state_dict(state)
    wrapped.to(device).eval()
    for p in wrapped.parameters():
        p.requires_grad_(False)
    return wrapped, mri_alphas_cumprod().to(device)
