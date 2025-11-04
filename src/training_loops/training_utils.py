import torch
from torchvision.utils import make_grid, save_image

## Inferencia desde T hasta 0 haciendo denoising ##
@torch.no_grad()
def sample_ddpm(
    model, diffusion, n: int, img_size: int = 64, device="cuda",
    steps: int = None,
    save_path: str = None, return_grid: bool = True):

    model.eval()
    T = diffusion.T if steps is None else steps
    x = torch.randn(n, 3, img_size, img_size, device=device) # Vector aleatorio

    for i in reversed(range(T)):
        t = torch.full((n,), i, device=device, dtype=torch.long)

        def model_eps(x_t, t_t):
            return model(x_t, t_t)
        x = diffusion.p_sample_step(model_eps, x, t) # Quitamos un paso de noise

    x = x.clamp(-1, 1)
    x = (x + 1) * 0.5  # [0,1]
    grid = make_grid(x, nrow=int(n**0.5), padding=2)
    if save_path is not None:
        save_image(grid, save_path)

    return grid if return_grid else x


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]

def lr_warmup(optimizer, base_lr, step, warmup_steps=1000):
    """Warmup lineal por pasos (globales)."""
    if warmup_steps is None or warmup_steps <= 0:
        return
    lr = base_lr * min(1.0, (step + 1) / warmup_steps)
    for g in optimizer.param_groups:
        g["lr"] = lr

@torch.no_grad()
def _swap_to_ema_and_sample(
    model, ema, diffusion, sample_fn, sample_n, img_size, device, out_path):

    """Copia EMA → modelo, hace muestras y restaura pesos online."""
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    _ = sample_fn(model, diffusion, n=sample_n, img_size=img_size,
                  device=device, save_path=out_path)
    model.load_state_dict(backup)


def compute_grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            total += float(p.grad.detach().pow(2).sum().item())
    return total ** 0.5


def gpu_mem_mb(device="cuda"):
    if torch.cuda.is_available() and device == "cuda":
        alloc = torch.cuda.memory_allocated() / (1024**2)
        reserv = torch.cuda.memory_reserved() / (1024**2)
        return alloc, reserv
    return 0.0, 0.0