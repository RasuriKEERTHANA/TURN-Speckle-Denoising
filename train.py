import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time

# =========================
# LOAD DATASET
# =========================
def load_dataset(folder_path, size=256, limit=80):
    images = []
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png') or filename.endswith('.jpg'):
            img = Image.open(os.path.join(folder_path, filename))
            img = img.convert('L')
            img = img.resize((size, size))
            img = np.array(img) / 255.0
            tensor = torch.tensor(img, dtype=torch.float32)
            tensor = tensor.unsqueeze(0).unsqueeze(0)
            images.append(tensor)
            if len(images) >= limit:
                break
    print(f"Loaded {len(images)} images")
    return images

# =========================
# GAUSSIAN KERNEL
# =========================
def get_gaussian_kernel():
    k = np.array([
        [1,4,7,4,1],
        [4,16,26,16,4],
        [7,26,41,26,7],
        [4,16,26,16,4],
        [1,4,7,4,1]
    ], dtype=np.float32)
    k /= k.sum()
    return torch.tensor(k).unsqueeze(0).unsqueeze(0)

# =========================
# GAMMA NOISE
# =========================
def add_gamma_noise(img, L):
    concentration = torch.tensor(float(L))
    rate = torch.tensor(float(L))
    noise = torch.distributions.Gamma(concentration, rate).sample(img.shape)
    return torch.clamp(img * noise, 1e-6, 1)

# =========================
# METRICS
# =========================
def psnr(x, y):
    mse = torch.mean((x - y)**2)
    return 10 * torch.log10(1.0 / mse)

def ssim(x, y):
    C1 = 0.01**2
    C2 = 0.03**2
    mu_x = torch.mean(x)
    mu_y = torch.mean(y)
    sigma_x = torch.var(x)
    sigma_y = torch.var(y)
    sigma_xy = torch.mean((x - mu_x)*(y - mu_y))
    return ((2*mu_x*mu_y + C1)*(2*sigma_xy + C2)) / \
           ((mu_x**2 + mu_y**2 + C1)*(sigma_x + sigma_y + C2))

# =========================
# GRADIENT & DIVERGENCE
# =========================
def gradient(u):
    ux = u[:,:,:,1:] - u[:,:,:,:-1]
    uy = u[:,:,1:,:] - u[:,:,:-1,:]
    ux = torch.cat([ux, torch.zeros_like(ux[:,:,:,:1])], 3)
    uy = torch.cat([uy, torch.zeros_like(uy[:,:,:1,:])], 2)
    return ux, uy

def divergence(px, py):
    px = torch.cat([px[:,:,:,:1], px], 3)
    py = torch.cat([py[:,:,:1,:], py], 2)
    return (px[:,:,:,1:]-px[:,:,:,:-1]) + (py[:,:,1:,:]-py[:,:,:-1,:])

# =========================
# STAGE
# =========================
class Stage(nn.Module):
    def __init__(self, gamma=0.1):
        super().__init__()
        self.nu      = nn.Parameter(torch.tensor(1.0))
        self.k       = nn.Parameter(torch.tensor(0.2))
        self.lambda_ = nn.Parameter(torch.tensor(1.0))
        self.gamma   = gamma
        self.dt      = nn.Parameter(torch.tensor(0.3))
        self.smooth  = nn.Conv2d(1, 1, 5, padding=2, bias=False)
        self.smooth.weight.data = get_gaussian_kernel()
        self.smooth.weight.requires_grad = False

    def forward(self, u, u_prev, u0):
        u_s = self.smooth(u)
        ux, uy = gradient(u_s)
        grad = torch.sqrt(ux**2 + uy**2 + 1e-6)
        M = torch.max(torch.abs(u_s))
        s = torch.abs(u_s) / (M + 1e-6)
        g = 2*(s**self.nu) / (1 + s**self.nu)
        c = 1 / (1 + (grad/(self.k+1e-6))**2)
        px = g * c * ux
        py = g * c * uy
        div = divergence(px, py)
        u_new = ((2+self.gamma*self.dt)*u - u_prev +
                 (self.dt**2)*self.lambda_*div) / (1+self.gamma*self.dt)
        u_new = u_new + self.lambda_*(u0 - u)
        return torch.clamp(u_new, -2, 2)

# =========================
# FULL TURN MODEL
# =========================
class TURN(nn.Module):
    def __init__(self, stages=25):
        super().__init__()
        self.stages = nn.ModuleList([Stage() for _ in range(stages)])

    def forward(self, x):
        u0     = x
        u_prev = x
        u      = x
        for s in self.stages:
            u_new  = s(u, u_prev, u0)
            u_prev = u
            u      = u_new
        return u

# =========================
# PDE BASELINE
# =========================
def classical_pde(noisy_log, steps=15):
    u      = noisy_log.clone()
    u_prev = noisy_log.clone()
    gamma  = 0.1
    for _ in range(steps):
        ux, uy = gradient(u)
        grad   = torch.sqrt(ux**2 + uy**2 + 1e-6)
        c      = 1 / (1 + grad**2)
        px     = c * ux
        py     = c * uy
        div    = divergence(px, py)
        u_new  = u + 0.3*div - gamma*(u - u_prev)
        u_prev = u
        u      = u_new
    return u

# =========================
# TRAINING
# =========================
def train_model(dataset, L, epochs=80):
    model = TURN()
    opt   = optim.Adam(model.parameters(), lr=1e-4)

    print(f"\n===== TRAINING L={L} =====")

    for epoch in range(epochs):
        total_loss = 0

        for clean in dataset:
            noisy     = add_gamma_noise(clean, L)
            noisy_log = torch.log(torch.clamp(noisy, min=1e-6))
            out_log   = model(noisy_log)
            out = torch.clamp(torch.exp(torch.clamp(out_log, -10, 10)), 0, 1)

            loss = torch.mean((out - clean)**2)
            if torch.isnan(loss):
                continue
            loss += 0.3 * torch.mean(torch.abs(out - clean))
            ux, uy = gradient(out)
            cx, cy = gradient(clean)
            loss += 0.2 * torch.mean(torch.abs(ux-cx) + torch.abs(uy-cy))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataset)
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{epochs} | Loss: {avg_loss:.4f}")

    print(f"Training complete for L={L}")
    return model

# =========================
# EVALUATE
# =========================
def evaluate(model, dataset, L):
    model.eval()
    total_psnr = 0
    total_ssim = 0

    with torch.no_grad():
        for clean in dataset:
            noisy     = add_gamma_noise(clean, L)
            noisy_log = torch.log(torch.clamp(noisy, min=1e-6))
            out_log   = model(noisy_log)
            out = torch.clamp(torch.exp(torch.clamp(out_log, -10, 10)), 0, 1)
            total_psnr += psnr(out, clean).item()
            total_ssim += ssim(out, clean).item()

    avg_psnr = total_psnr / len(dataset)
    avg_ssim = total_ssim / len(dataset)
    print(f"L={L} | PSNR: {avg_psnr:.2f} dB | SSIM: {avg_ssim:.4f}")
    return avg_psnr, avg_ssim

# =========================
# VISUALIZE
# =========================
def visualize(model, clean, L, idx=0, save_dir='results'):
    model.eval()
    os.makedirs(save_dir, exist_ok=True)
    with torch.no_grad():
        noisy     = add_gamma_noise(clean, L)
        noisy_log = torch.log(torch.clamp(noisy, min=1e-6))
        out_log   = model(noisy_log)
        out = torch.clamp(torch.exp(torch.clamp(out_log, -10, 10)), 0, 1)

        pde_log = classical_pde(noisy_log)
        pde_out = torch.clamp(torch.exp(pde_log), 0, 1)

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    axes[0].imshow(clean.squeeze(),   cmap='gray'); axes[0].set_title('Clean')
    axes[1].imshow(noisy.squeeze(),   cmap='gray'); axes[1].set_title(f'Noisy L={L}')
    axes[2].imshow(out.squeeze(),     cmap='gray'); axes[2].set_title('TURN (Ours)')
    axes[3].imshow(pde_out.squeeze(), cmap='gray'); axes[3].set_title('PDE Baseline')
    plt.tight_layout()
    save_path = os.path.join(save_dir, f'result_L{L}_img{idx}.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Saved {save_path}")

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='dataset/BSD400',
                        help='Path to training images folder')
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--limit', type=int, default=80)
    parser.add_argument('--save_dir', type=str, default='models')
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load dataset
    dataset = load_dataset(args.data, limit=args.limit)

    # Train
    model_L1  = train_model(dataset, L=1,  epochs=args.epochs)
    model_L10 = train_model(dataset, L=10, epochs=args.epochs)

    # Save models
    torch.save(model_L1.state_dict(),  os.path.join(args.save_dir, 'model_L1.pth'))
    torch.save(model_L10.state_dict(), os.path.join(args.save_dir, 'model_L10.pth'))
    print("Models saved.")

    # Evaluate
    print("\n===== FINAL RESULTS =====")
    psnr_L1,  ssim_L1  = evaluate(model_L1,  dataset, L=1)
    psnr_L10, ssim_L10 = evaluate(model_L10, dataset, L=10)
    print(f"\nL=1  → PSNR: {psnr_L1:.2f} dB  | SSIM: {ssim_L1:.4f}")
    print(f"L=10 → PSNR: {psnr_L10:.2f} dB | SSIM: {ssim_L10:.4f}")

    # Visualize sample images
    for idx in [0, 10, 20]:
        visualize(model_L1,  dataset[idx], L=1,  idx=idx)
        visualize(model_L10, dataset[idx], L=10, idx=idx)
