"""
DDPM for custom image denoising.
"""

import os
import math
import glob
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#Time embedding
class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = time[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


# U-Net building blocks
class ResidualBlock(nn.Module):
    """Residual block with time-embedding injection."""

    def __init__(self, in_ch, out_ch, time_emb_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.norm2 = nn.GroupNorm(8, out_ch)

        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, out_ch),
        )
        self.residual_conv = (
            nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        )

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)

        # Add time embedding
        t = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t

        h = self.conv2(h)
        h = self.norm2(h)
        h = F.silu(h)

        return h + self.residual_conv(x)


class SelfAttention(nn.Module):
    """Simple single-head self-attention."""

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        self.scale = channels ** -0.5

    def forward(self, x):
        b, c, h, w = x.shape
        x_norm = self.norm(x)
        qkv = self.qkv(x_norm).reshape(b, 3, c, h * w)
        q, k, v = qkv[:, 0], qkv[:, 1], qkv[:, 2]  # (b, c, hw)

        attn = torch.einsum("bci,bcj->bij", q * self.scale, k)
        attn = attn.softmax(dim=-1)
        out = torch.einsum("bij,bcj->bci", attn, v)
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)


# U-Net model

class UNet(nn.Module):
    """
    Input  : (B, in_channels, H, W)  — noisy image
    Time t : (B,) long tensor
    Output : (B, in_channels, H, W)  — predicted noise
    """

    def __init__(self, in_channels=3, base_channels=64, time_emb_dim=256):
        super().__init__()

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4

        # ---- Encoder ----
        self.enc1 = ResidualBlock(in_channels, c1, time_emb_dim)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualBlock(c1, c2, time_emb_dim)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualBlock(c2, c3, time_emb_dim)
        self.attn = SelfAttention(c3)

        # ---- Bottleneck ----
        self.mid = ResidualBlock(c3, c3, time_emb_dim)

        # ---- Decoder ----
        self.up2 = nn.ConvTranspose2d(c3, c3, 2, stride=2)
        self.dec2 = ResidualBlock(c3 + c2, c2, time_emb_dim)

        self.up1 = nn.ConvTranspose2d(c2, c2, 2, stride=2)
        self.dec1 = ResidualBlock(c2 + c1, c1, time_emb_dim)

        self.out = nn.Conv2d(c1, in_channels, 1)

    def forward(self, x, t):
        t_emb = self.time_mlp(t)

        # Encoder
        e1 = self.enc1(x, t_emb)              # H
        e2 = self.enc2(self.pool1(e1), t_emb) # H/2
        e3 = self.enc3(self.pool2(e2), t_emb) # H/4
        e3 = self.attn(e3)

        # Bottleneck
        m = self.mid(e3, t_emb)

        # Decoder
        d2 = self.up2(m)
        d2 = self.dec2(torch.cat([d2, e2], dim=1), t_emb)

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([d1, e1], dim=1), t_emb)

        return self.out(d1)



# DDPM scheduler
class DDPMScheduler:
    def __init__(self, num_timesteps=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.num_timesteps = num_timesteps
        self.device = device

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )

    def q_sample(self, x0, t, noise=None):
        """Forward process: x_t = sqrt(a_bar)*x0 + sqrt(1-a_bar)*eps"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_1ma = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return sqrt_a * x0 + sqrt_1ma * noise, noise

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        """One reverse step: p(x_{t-1} | x_t)."""
        t_batch = torch.full((x_t.size(0),), t, device=self.device, dtype=torch.long)
        eps_pred = model(x_t, t_batch)

        beta_t = self.betas[t]
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alphas_cumprod[t]

        mean = (1.0 / torch.sqrt(alpha_t)) * (x_t - (beta_t / torch.sqrt(1.0 - alpha_bar_t)) * eps_pred)

        if t == 0:
            return mean

        var = self.posterior_variance[t]
        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    @torch.no_grad()
    def sample(self, model, shape, save_every=None, save_dir=None):
        """Full reverse diffusion (generation)."""
        model.eval()
        x = torch.randn(shape, device=self.device)
        for t in tqdm(reversed(range(self.num_timesteps)), desc="Sampling"):
            x = self.p_sample(model, x, t)
            if save_every and t % save_every == 0 and save_dir:
                save_image(x.clamp(-1, 1) * 0.5 + 0.5, f"{save_dir}/step_{t:04d}.png")
        return x

    @torch.no_grad()
    def denoise(self, model, noisy_image, start_t):
        """
        Reverse diffusion
        """
        model.eval()
        x = noisy_image.to(self.device)
        for t in tqdm(reversed(range(start_t + 1)), desc="Denoising"):
            x = self.p_sample(model, x, t)
        return x

#dataset
class CustomImageDataset(Dataset):
    def __init__(self, folder, image_size=64, exts=("*.png", "*.jpg", "*.jpeg")):
        self.paths = []
        for ext in exts:
            self.paths.extend(glob.glob(os.path.join(folder, "**", ext), recursive=True))
        self.paths = sorted(self.paths)
        if not self.paths:
            raise RuntimeError(f"No images found in {folder}")

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),             # [0,1]
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # [-1,1]
        ])
        print(f"[Dataset] Found {len(self.paths)} images in {folder}")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

#Training loop
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    dataset = CustomImageDataset(args.data_dir, image_size=args.image_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
    num_workers=args.num_workers, drop_last=True,)

    model = UNet(in_channels=3, base_channels=args.base_channels).to(device)
    scheduler = DDPMScheduler(num_timesteps=args.timesteps, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda")

    os.makedirs(args.out_dir, exist_ok=True)

    global_step = 0
    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x0 in pbar:
            x0 = x0.to(device)
            b = x0.size(0)
            t = torch.randint(0, args.timesteps, (b,), device=device).long()

            with torch.amp.autocast("cuda"):
                x_t, noise = scheduler.q_sample(x0, t)
                eps_pred = model(x_t, t)
                loss = F.mse_loss(eps_pred, noise)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            pbar.set_postfix(loss=f"{loss.item():.4f}")
            global_step += 1

        # Save checkpoint & a sample
        if (epoch + 1) % args.save_every == 0 or epoch == args.epochs - 1:
            ckpt = {
                "model": model.state_dict(),
                "epoch": epoch,
                "args": vars(args),
            }
            torch.save(ckpt, os.path.join(args.out_dir, f"ckpt_epoch{epoch+1}.pt"))
            print(f"[Saved] {args.out_dir}/ckpt_epoch{epoch+1}.pt")

            if args.sample_during_train:
                samples = scheduler.sample(model, (4, 3, args.image_size, args.image_size))
                save_image(
                    samples.clamp(-1, 1) * 0.5 + 0.5,
                    os.path.join(args.out_dir, f"sample_epoch{epoch+1}.png"),
                    nrow=2,
                )

#Denoise a custom image (inference)
@torch.no_grad()
def denoise_image(args):
    ckpt = torch.load(args.ckpt, map_location=device)
    cfg = ckpt.get("args", {})
    image_size = cfg.get("image_size", args.image_size)
    base_channels = cfg.get("base_channels", args.base_channels)
    timesteps = cfg.get("timesteps", args.timesteps)

    model = UNet(in_channels=3, base_channels=base_channels).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    scheduler = DDPMScheduler(num_timesteps=timesteps, device=device)

    # Load input image -> [-1,1]
    tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])
    img = Image.open(args.input).convert("RGB")
    x0 = tf(img).unsqueeze(0).to(device)

    # If the user supplied a "noisy" image, we can first project it into the
    # diffusion chain by adding noise up to `start_t`.
    start_t = args.start_t if args.start_t is not None else args.timesteps - 1
    if args.input_is_clean:
        t = torch.tensor([start_t], device=device)
        x_start, _ = scheduler.q_sample(x0, t)
    else:
        x_start = x0  # treat input as already noisy

    # Save the noisy input 
    os.makedirs(args.out_dir, exist_ok=True)
    save_image(
        (x_start.clamp(-1, 1) * 0.5 + 0.5),
        os.path.join(args.out_dir, "input_noisy.png"),
    )

    # Reverse diffusion
    denoised = scheduler.denoise(model, x_start, start_t)

    save_image(
        (denoised.clamp(-1, 1) * 0.5 + 0.5),
        os.path.join(args.out_dir, "denoised.png"),
    )

    # save a side-by-side comparison
    comparison = torch.cat([x_start, denoised], dim=0)
    save_image(
        (comparison.clamp(-1, 1) * 0.5 + 0.5),
        os.path.join(args.out_dir, "comparison.png"),
        nrow=2,
    )
    print(f"[Saved] {args.out_dir}/denoised.png and comparison.png")


#Main Loop
def build_parser():
    p = argparse.ArgumentParser("DDPM for Laryngeal image denoising")
    sub = p.add_subparsers(dest="mode", required=True)

    # ---- train ----
    tr = sub.add_parser("train", help="Train DDPM on a Laryngeal image folder")
    tr.add_argument("--data_dir", type=str, required=True, help="Folder containing")
    tr.add_argument("--out_dir", type=str, default="/data")
    tr.add_argument("--image_size", type=int, default=256)
    tr.add_argument("--batch_size", type=int, default=8)
    tr.add_argument("--epochs", type=int, default=50)
    tr.add_argument("--lr", type=float, default=2e-4)
    tr.add_argument("--timesteps", type=int, default=1000)
    tr.add_argument("--base_channels", type=int, default=64)
    tr.add_argument("--num_workers", type=int, default=2)
    tr.add_argument("--save_every", type=int, default=5)
    tr.add_argument("--sample_during_train", action="store_true")

    # ---- denoise ----
    dn = sub.add_parser("denoise", help="Denoise an image with a trained model")
    dn.add_argument("--ckpt", type=str, required=True)
    dn.add_argument("--input", type=str, required=True,
                    help="Path to input image (noisy or clean)")
    dn.add_argument("--out_dir", type=str, default="runs/denoise")
    dn.add_argument("--image_size", type=int, default=64)
    dn.add_argument("--base_channels", type=int, default=64)
    dn.add_argument("--timesteps", type=int, default=1000)
    dn.add_argument("--start_t", type=int, default=None,  help="Starting timestep for reverse diffusion")
    dn.add_argument("--input_is_clean", action="store_true", help="If set, the input is treated as clean")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "denoise":
        denoise_image(args)


if __name__ == "__main__":
    main()
