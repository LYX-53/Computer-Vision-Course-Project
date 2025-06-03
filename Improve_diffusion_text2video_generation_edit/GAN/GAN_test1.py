import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
from ..utils.helper_functions import default, exists, prob_mask_like, is_list_str
from ..text.text_handler import bert_embed, tokenize
from einops_exts import check_shape
from tqdm import tqdm


class GANGenerator(nn.Module):
    """
    GAN Generator that replaces the diffusion model's denoising function.
    """
4
    def __init__(self, denoise_fn):
        super().__init__()
        self.denoise_fn = denoise_fn  # Reusing the same architecture but as a generator

    def forward(self, noise, cond=None, cond_scale=1.0):
        # Generate images from random noise
        return self.denoise_fn(noise, torch.zeros_like(noise[:, 0]).long(), cond=cond, cond_scale=cond_scale)


class GANDiscriminator(nn.Module):
    """
    GAN Discriminator that classifies real vs fake images.
    """

    def __init__(self, channels=3, image_size=64, num_frames=16):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames

        # Simple discriminator architecture
        self.main = nn.Sequential(
            # input is (channels) x (num_frames) x (image_size) x (image_size)
            nn.Conv3d(channels, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm3d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv3d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, input):
        return self.main(input)


class GANModel(nn.Module):
    """
    GAN model that replaces the GaussianDiffusion model.
    """

    def __init__(
            self,
            denoise_fn,
            *,
            image_size,
            num_frames,
            text_use_bert_cls=False,
            channels=3,
            loss_type='l1',  # Not used in GAN, kept for compatibility
            use_dynamic_thres=False,
            dynamic_thres_percentile=0.9
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames

        # Create generator and discriminator
        self.generator = GANGenerator(denoise_fn)
        self.discriminator = GANDiscriminator(channels, image_size, num_frames)

        # Loss function
        self.loss = nn.BCELoss()

        # Text conditioning parameters
        self.text_use_bert_cls = text_use_bert_cls

        # Dynamic thresholding (not typically used in GANs, kept for compatibility)
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def sample_noise(self, batch_size, device):
        """Generate random noise for the generator."""
        return torch.randn(batch_size, self.channels, self.num_frames,
                           self.image_size, self.image_size, device=device)

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1.0, batch_size=16):
        """
        Generate samples using the generator.
        """
        device = next(self.generator.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)

        batch_size = cond.shape[0] if exists(cond) else batch_size
        noise = self.sample_noise(batch_size, device)

        # Generate images
        generated = self.generator(noise, cond=cond, cond_scale=cond_scale)
        return (generated + 1) * 0.5  # Scale to [0, 1]

    def forward(self, x, cond=None, real_labels=None, fake_labels=None):
        """
        GAN forward pass that computes generator and discriminator losses.
        """
        device = x.device
        batch_size = x.shape[0]

        # Prepare labels
        real_labels = default(real_labels, torch.ones(batch_size, 1, device=device))
        fake_labels = default(fake_labels, torch.zeros(batch_size, 1, device=device))

        # Process text conditioning if provided
        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)

        # Train discriminator with real images
        self.discriminator.zero_grad()
        real_output = self.discriminator(x)
        d_loss_real = self.loss(real_output, real_labels)

        # Train discriminator with fake images
        noise = self.sample_noise(batch_size, device)
        fake_images = self.generator(noise, cond=cond)
        fake_output = self.discriminator(fake_images.detach())
        d_loss_fake = self.loss(fake_output, fake_labels)

        # Total discriminator loss
        d_loss = d_loss_real + d_loss_fake

        # Train generator
        self.generator.zero_grad()
        fake_output = self.discriminator(fake_images)
        g_loss = self.loss(fake_output, real_labels)  # Generator wants discriminator to think fakes are real

        return {
            'g_loss': g_loss,
            'd_loss': d_loss,
            'fake_images': fake_images,
            'real_scores': real_output,
            'fake_scores': fake_output
        }

    def p_losses(self, x, *args, **kwargs):
        """Alias for forward to maintain compatibility."""
        return self.forward(x, *args, **kwargs)

    # The following methods are kept for compatibility but don't do anything in GAN context
    def q_mean_variance(self, *args, **kwargs):
        pass

    def predict_start_from_noise(self, *args, **kwargs):
        pass

    def q_posterior(self, *args, **kwargs):
        pass

    def p_mean_variance(self, *args, **kwargs):
        pass

    def p_sample(self, *args, **kwargs):
        pass

    def p_sample_loop(self, *args, **kwargs):
        pass

    def q_sample(self, *args, **kwargs):
        pass

    def interpolate(self, *args, **kwargs):
        pass