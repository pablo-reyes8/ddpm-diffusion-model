import torch
import torch.nn as nn
import math
from src.model.difussion_class import * 
from src.model.unet_backbone import *


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())

def test_DDPM():
    torch.manual_seed(7)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_enabled = device.type == "cuda"

    # Instancias
    # U-Net: sin atención para MVP
    model = build_unet_64x64(attn_resolutions=set(), dropout=0.0)
    diffusion = Diffusion(T=1000, schedule="linear", beta_min=1e-4, beta_max=2e-2)

    model = model.to(device)
    diffusion = diffusion.to(device)

    print(f"[INFO] Device   : {device}")
    print(f"[INFO] Params   : {count_params(model):,} parámetros")
    print(f"[INFO] T steps  : {diffusion.T}")
    print()

    # Batch sintético
    B, C, H, W = 8, 3, 64, 64
    x0 = torch.empty(B, C, H, W, device=device).uniform_(-1.0, 1.0)
    t  = diffusion.sample_timesteps(B, device=device)

    # Forward shape + no NaNs
    with torch.no_grad():
        eps_pred = model(x0, t)
    assert eps_pred.shape == x0.shape, "La salida del U-Net no coincide con (B,3,64,64)"
    assert torch.isfinite(eps_pred).all(), "NaNs/Inf en la salida del U-Net"
    print("[OK] Forward U-Net: shapes y finitud correctos.")

    # L_simple y backward
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    opt.zero_grad(set_to_none=True)

    if amp_enabled:
        scaler = torch.cuda.amp.GradScaler()
        with torch.cuda.amp.autocast():
            loss = diffusion.loss_simple(model, x0, t)
        assert loss.ndim == 0, "La loss debe ser un escalar"
        scaler.scale(loss).backward()

        total_grad = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += float(p.grad.detach().abs().mean().item())
        assert math.isfinite(total_grad) and total_grad > 0.0, "Gradientes no finitos o ~0"
        scaler.step(opt); scaler.update()
    else:
        loss = diffusion.loss_simple(model, x0, t)
        assert loss.ndim == 0, "La loss debe ser un escalar"
        loss.backward()
        total_grad = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_grad += float(p.grad.detach().abs().mean().item())
        assert math.isfinite(total_grad) and total_grad > 0.0, "Gradientes no finitos o ~0"
        opt.step()

    print(f"[OK] Loss y backward: loss={float(loss):.5f}, grad|mean|≈{total_grad:.3e}")

    # q_sample estadísticas
    with torch.no_grad():
        t_hi = torch.full((B,), int(0.8 * diffusion.T), device=device, dtype=torch.long)
        t_lo = torch.full((B,), int(0.1 * diffusion.T), device=device, dtype=torch.long)

        x_hi = diffusion.q_sample(x0, t_hi)  
        x_lo = diffusion.q_sample(x0, t_lo) 

        def stats(z):
            return z.mean().item(), z.std().item(), z.min().item(), z.max().item()

        m_hi, s_hi, mn_hi, mx_hi = stats(x_hi)
        m_lo, s_lo, mn_lo, mx_lo = stats(x_lo)

        assert torch.isfinite(x_hi).all() and torch.isfinite(x_lo).all(), "NaNs/Inf en q_sample"
        print(f"[OK] q_sample stats:")
        print(f" t≈0.8T -> mean={m_hi:+.3f}, std={s_hi:.3f}, min={mn_hi:+.3f}, max={mx_hi:+.3f}")
        print(f" t≈0.1T -> mean={m_lo:+.3f}, std={s_lo:.3f}, min={mn_lo:+.3f}, max={mx_lo:+.3f}")

    # p_sample_step (DDPM 1 paso)
    with torch.no_grad():
        t_step = torch.full((B,), int(0.6 * diffusion.T), device=device, dtype=torch.long)
        x_t = diffusion.q_sample(x0, t_step)

        def model_eps_pred_fn(x, t):
            return model(x, t)  

        x_prev = diffusion.p_sample_step(model_eps_pred_fn, x_t, t_step)
        assert x_prev.shape == x_t.shape and torch.isfinite(x_prev).all(), "p_sample_step falla en shape/NaN"
        print("[OK] p_sample_step: un paso de muestreo DDPM estable.")

    model_attn = build_unet_64x64(attn_resolutions={16, 8}, dropout=0.0).to(device)

    with torch.no_grad():
        y = model_attn(x0, t)
    assert y.shape == x0.shape and torch.isfinite(y).all(), "U-Net+attn: shape/NaN"
    print("[OK] U-Net con atención en 16x16/8x8: shapes correctos.")

    print("\n[ALL GOOD] Sanity checks superados.")

