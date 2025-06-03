import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from ..utils.helper_functions import default, exists, prob_mask_like, is_list_str
from ..text.text_handler import bert_embed, tokenize
from einops_exts import check_shape
from tqdm import tqdm
import copy
from torchvision.models import vgg16_bn
from typing import Optional, Tuple


def extract(a: torch.Tensor, t: torch.Tensor, x_shape: torch.Size) -> torch.Tensor:
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class AdaptiveGroupNorm(nn.Module):
    """Improved adaptive group normalization with better conditioning"""

    def __init__(self, dim: int, time_emb_dim: int, cond_dim: int, num_groups: int = 32):
        super().__init__()
        self.num_groups = num_groups
        self.norm = nn.GroupNorm(num_groups=num_groups, num_channels=dim, eps=1e-6)

        # Time embedding processing
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim * 2)
        )

        # Condition processing with more capacity
        self.cond_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, dim * 2)
        )

        # Learnable scale factors
        self.time_scale = nn.Parameter(torch.ones(1))
        self.cond_scale = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor, cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Process time embedding
        time_scale_bias = self.time_mlp(time_emb) * self.time_scale

        # Process condition if provided
        if cond is not None:
            cond_scale_bias = self.cond_mlp(cond) * self.cond_scale
            scale_bias = time_scale_bias + cond_scale_bias
        else:
            scale_bias = time_scale_bias

        scale, bias = scale_bias.chunk(2, dim=1)

        # Apply group norm and adaptive scaling
        x = self.norm(x)
        return x * (scale.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) + bias.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1))


class GaussianDiffusion(nn.Module):
    def __init__(
            self,
            denoise_fn: nn.Module,
            *,
            image_size: int,
            num_frames: int,
            text_use_bert_cls: bool = False,
            channels: int = 3,
            timesteps: int = 1000,
            loss_type: str = 'l1',
            use_dynamic_thres: bool = False,
            dynamic_thres_percentile: float = 0.9,
            # Improved parameters
            use_ema: bool = True,
            ema_decay: float = 0.9999,
            ema_update_every: int = 10,  # Only update EMA every N steps
            use_perceptual_loss: bool = True,
            perceptual_loss_weight: float = 0.1,
            perceptual_layers: Tuple[int] = (3, 8, 15, 22),  # Multiple layers for perceptual loss
            ddim_sampling_eta: float = 0.0,
            clip_x_start: bool = True,  # Whether to clip predicted x_start
    ) -> None:
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.clip_x_start = clip_x_start

        # Improved EMA initialization
        self.use_ema = use_ema
        self.ema_decay = ema_decay
        self.ema_update_every = ema_update_every
        self.ema_update_count = 0
        if use_ema:
            self.ema_denoise_fn = copy.deepcopy(denoise_fn)
            for param in self.ema_denoise_fn.parameters():
                param.requires_grad_(False)

        # Enhanced perceptual loss
        self.use_perceptual_loss = use_perceptual_loss
        self.perceptual_loss_weight = perceptual_loss_weight
        self.perceptual_layers = perceptual_layers
        if use_perceptual_loss:
            self.vgg = self._build_vgg_features()
            for param in self.vgg.parameters():
                param.requires_grad = False
            self.vgg.eval()

        # DDIM parameters
        self.ddim_sampling_eta = ddim_sampling_eta

        # Original diffusion setup
        betas = cosine_beta_schedule(timesteps)
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))
        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # Diffusion calculations
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod', torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1))

        # Posterior calculations
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance)
        register_buffer('posterior_log_variance_clipped', torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer('posterior_mean_coef1', betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod))

        self.text_use_bert_cls = text_use_bert_cls
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def _build_vgg_features(self) -> nn.Module:
        """Build VGG feature extractor for perceptual loss"""
        vgg = vgg16_bn(pretrained=True).features
        layers = []
        for i, layer in enumerate(vgg):
            layers.append(layer)
            if i == max(self.perceptual_layers):
                break
        return nn.Sequential(*layers)

    def update_ema(self):
        """Improved EMA update with step counting"""
        if not self.use_ema:
            return

        self.ema_update_count += 1
        if self.ema_update_count % self.ema_update_every != 0:
            return

        with torch.no_grad():
            decay = min(self.ema_decay, (1 + self.ema_update_count) / (10 + self.ema_update_count))
            for ema_param, param in zip(self.ema_denoise_fn.parameters(), self.denoise_fn.parameters()):
                ema_param.data.mul_(decay).add_(param.data, alpha=1 - decay)

    def q_mean_variance(self, x_start: torch.Tensor, t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1. - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor]:
        posterior_mean = (
                extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
                extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            clip_denoised: bool,
            cond: Optional[torch.Tensor] = None,
            cond_scale: float = 1.0,
            denoise_fn: Optional[nn.Module] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        denoise_fn = denoise_fn or self.denoise_fn
        model_output = denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        x_recon = self.predict_start_from_noise(x, t=t, noise=model_output)

        if clip_denoised or self.clip_x_start:
            s = 1.
            if self.use_dynamic_thres:
                s = torch.quantile(
                    rearrange(x_recon, 'b ... -> b (...)').abs(),
                    self.dynamic_thres_percentile,
                    dim=-1
                ).clamp_(min=1.).view(-1, *((1,) * (x_recon.ndim - 1)))
            x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_recon, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(
            self,
            x: torch.Tensor,
            t: torch.Tensor,
            cond: Optional[torch.Tensor] = None,
            cond_scale: float = 1.0,
            clip_denoised: bool = True,
            denoise_fn: Optional[nn.Module] = None,
            return_pred_xstart: bool = False
    ) -> torch.Tensor:
        denoise_fn = denoise_fn or self.denoise_fn
        b, *_, device = *x.shape, x.device

        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised,
            cond=cond, cond_scale=cond_scale, denoise_fn=denoise_fn
        )

        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))

        sample = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        if return_pred_xstart:
            return sample, model_mean
        return sample

    @torch.inference_mode()
    def p_sample_loop(
            self,
            shape: torch.Size,
            cond: Optional[torch.Tensor] = None,
            cond_scale: float = 1.0,
            denoise_fn: Optional[nn.Module] = None,
            ddim_num_steps: Optional[int] = None,
            progress: bool = True
    ) -> torch.Tensor:
        device = self.betas.device
        b = shape[0]
        img = torch.randn(shape, device=device)
        denoise_fn = denoise_fn or self.denoise_fn

        # DDIM sampling
        if ddim_num_steps is not None:
            time_seq = list(reversed(range(0, self.num_timesteps, self.num_timesteps // ddim_num_steps)))
        else:
            time_seq = reversed(range(0, self.num_timesteps))

        iterator = tqdm(time_seq, desc='sampling loop time step') if progress else time_seq

        for i in iterator:
            times = torch.full((b,), i, device=device, dtype=torch.long)
            img = self.p_sample(img, times, cond=cond, cond_scale=cond_scale, denoise_fn=denoise_fn)

        return (img + 1) * 0.5  # Scale from [-1, 1] to [0, 1]

    @torch.inference_mode()
    def sample(
            self,
            cond: Optional[torch.Tensor] = None,
            cond_scale: float = 1.0,
            batch_size: int = 16,
            use_ema: bool = True,
            ddim_num_steps: Optional[int] = None,
            progress: bool = True
    ) -> torch.Tensor:
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        denoise_fn = self.ema_denoise_fn if (use_ema and self.use_ema) else self.denoise_fn

        return self.p_sample_loop(
            (batch_size, self.channels, self.num_frames, self.image_size, self.image_size),
            cond=cond, cond_scale=cond_scale, denoise_fn=denoise_fn,
            ddim_num_steps=ddim_num_steps, progress=progress
        )

    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
                extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def _compute_perceptual_loss(self, x_start: torch.Tensor, x_recon: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss using multiple VGG layers"""
        x_start_norm = (x_start + 1) * 0.5  # [-1,1] -> [0,1]
        x_recon_norm = (x_recon + 1) * 0.5

        # Randomly select frames for efficiency
        b, c, f, h, w = x_start.shape
        frame_idx = torch.randint(0, f, (1,)).item()
        x_start_frame = x_start_norm[:, :, frame_idx]
        x_recon_frame = x_recon_norm[:, :, frame_idx]

        # Compute features at multiple layers
        loss = 0
        x_start_feat = x_start_frame
        x_recon_feat = x_recon_frame
        for i, layer in enumerate(self.vgg):
            x_start_feat = layer(x_start_feat)
            x_recon_feat = layer(x_recon_feat)
            if i in self.perceptual_layers:
                loss = loss + F.mse_loss(x_recon_feat, x_start_feat.detach())

        return loss / len(self.perceptual_layers)

    def p_losses(
            self,
            x_start: torch.Tensor,
            t: torch.Tensor,
            cond: Optional[torch.Tensor] = None,
            noise: Optional[torch.Tensor] = None,
            **kwargs
    ) -> torch.Tensor:
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls).to(device)

        x_recon = self.denoise_fn(x_noisy, t, cond=cond, **kwargs)

        # Base loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        # Add perceptual loss if enabled
        if self.use_perceptual_loss:
            perceptual_loss = self._compute_perceptual_loss(x_start, x_recon)
            loss = loss + self.perceptual_loss_weight * perceptual_loss

        return loss

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        b, device = x.shape[0], x.device
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames, h=self.image_size, w=self.image_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        x = (x * 2) - 1  # Scale from [0,1] to [-1,1]
        loss = self.p_losses(x, t, *args, **kwargs)

        # Update EMA if enabled
        if self.use_ema:
            self.update_ema()

        return loss