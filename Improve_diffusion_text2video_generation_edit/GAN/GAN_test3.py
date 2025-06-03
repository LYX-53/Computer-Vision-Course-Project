import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from ..utils.helper_functions import default, exists, prob_mask_like, is_list_str
from ..text.text_handler import bert_embed, tokenize
from einops_exts import check_shape
from tqdm import tqdm


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
            dynamic_thres_percentile: float = 0.9
    ) -> None:
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.loss_type = loss_type

        # Build discriminator with proper dimensions
        self.discriminator = nn.Sequential(
            nn.Conv3d(in_channels=channels, out_channels=64, kernel_size=(1, 3, 3)),  # (t, h, w)
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=(1, 3, 3)),
            nn.InstanceNorm3d(128),
            nn.LeakyReLU(0.2),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=(1, 3, 3)),
            nn.InstanceNorm3d(256),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.text_use_bert_cls = text_use_bert_cls
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

        # Dummy buffers for compatibility
        self.register_buffer('betas', torch.zeros(timesteps))
        self.register_buffer('alphas_cumprod', torch.zeros(timesteps))
        self.register_buffer('alphas_cumprod_prev', torch.zeros(timesteps))
        self.num_timesteps = timesteps

        # Add counter for tracking iterations
        self.register_buffer('_iter_count', torch.tensor(0))

    def sample_noise(self, batch_size, device):
        return torch.randn(batch_size, self.channels, self.num_frames,
                           self.image_size, self.image_size, device=device)

    def q_mean_variance(self, x_start, t):
        return x_start, torch.zeros_like(x_start), torch.zeros_like(x_start)

    def predict_start_from_noise(self, x_t, t, noise):
        return noise

    def q_posterior(self, x_start, x_t, t):
        return x_start, torch.zeros_like(x_start), torch.zeros_like(x_start)

    def p_mean_variance(self, x, t, clip_denoised, cond=None, cond_scale=1.0):
        noise = self.sample_noise(x.shape[0], x.device)
        x_recon = self.denoise_fn.forward_with_cond_scale(noise, t, cond=cond, cond_scale=cond_scale)
        return x_recon, torch.zeros_like(x_recon), torch.zeros_like(x_recon)

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1.0, clip_denoised=True):
        noise = self.sample_noise(x.shape[0], x.device)
        return self.denoise_fn.forward_with_cond_scale(noise, t, cond=cond, cond_scale=cond_scale)

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.0):
        noise = self.sample_noise(shape[0], self.betas.device)
        t = torch.zeros(shape[0], device=noise.device, dtype=torch.long)
        return self.denoise_fn.forward_with_cond_scale(noise, t, cond=cond, cond_scale=cond_scale)

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1.0, batch_size=16):
        device = next(self.denoise_fn.parameters()).device
        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)
        batch_size = cond.shape[0] if exists(cond) else batch_size
        return self.p_sample_loop((batch_size, self.channels, self.num_frames,
                                   self.image_size, self.image_size),
                                  cond=cond, cond_scale=cond_scale)

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return noise

    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        b, device = x_start.shape[0], x_start.device

        real_labels = torch.ones(b, 1, device=device)
        fake_labels = torch.zeros(b, 1, device=device)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)

        noise = default(noise, lambda: self.sample_noise(b, device))
        fake_images = self.denoise_fn(noise, t, cond=cond, **kwargs)

        real_out = self.discriminator(x_start)
        fake_out = self.discriminator(fake_images.detach())

        if self.loss_type == 'l1':
            d_loss = F.l1_loss(real_out, real_labels) + F.l1_loss(fake_out, fake_labels)
            g_loss = F.l1_loss(self.discriminator(fake_images), real_labels)
        elif self.loss_type == 'l2':
            d_loss = F.mse_loss(real_out, real_labels) + F.mse_loss(fake_out, fake_labels)
            g_loss = F.mse_loss(self.discriminator(fake_images), real_labels)
        else:
            d_loss = F.binary_cross_entropy(real_out, real_labels) + \
                     F.binary_cross_entropy(fake_out, fake_labels)
            g_loss = F.binary_cross_entropy(self.discriminator(fake_images), real_labels)

        # Increment iteration counter and print loss every 1000 steps
        self._iter_count += 1
        if self._iter_count % 1000 == 0:
            print(f'Step {self._iter_count.item()}: G Loss: {g_loss.item():.4f}, D Loss: {d_loss.item():.4f}')

        return g_loss + d_loss

    def forward(self, x, *args, **kwargs):
        b, device = x.shape[0], x.device
        check_shape(x, 'b c f h w', c=self.channels, f=self.num_frames,
                    h=self.image_size, w=self.image_size)
        x = (x * 2) - 1
        t = torch.zeros(b, dtype=torch.long, device=device)
        return self.p_losses(x, t, *args, **kwargs)

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        return (1 - lam) * x1 + lam * x2