import torch
from torchvision.utils import make_grid, save_image
from torchvision.utils import save_image as _tv_save_image

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


####### DDIM INFERENCE FOR TRAINING (LOW GPU USE) ############
def save_image_grid(x: torch.Tensor, path: str, nrow: int | None = None):
    """
    Guarda un grid de imágenes en `path`. Crea la carpeta únicamente si existe un directorio en la ruta.
    """
    x = x.detach().float().cpu()

    if nrow is None:
        n = x.size(0)
        nrow = int(n**0.5)
        if nrow < 1: 
            nrow = 1

    dirpath = os.path.dirname(path)
    if dirpath:      
        os.makedirs(dirpath, exist_ok=True)

    _tv_save_image(x, path, nrow=nrow)
    print(f"[OK] Guardado grid en {path}")


@torch.no_grad()
def ddim_sample(
    model, diffusion, *, n=16, img_size=256, device="cuda",
    ema=None, save_path=None, seed=1234,
    steps=50, eta=0.0, schedule="karras",  # "linear" | "cosine_alpha_bar" | "karras"
    clip_x0=True):
  
    was_training = model.training
    model.eval()
    backup = None
    if ema is not None:
        backup = {k: v.detach().clone() for k,v in model.state_dict().items()}
        ema.copy_to(model)

    if seed is not None:
        torch.manual_seed(seed)

    x = torch.randn(n, 3, img_size, img_size, device=device)
    T = diffusion.T

    if schedule == "linear":
        idx = torch.linspace(T-1, 0, steps+1, device=device)
    elif schedule == "cosine_alpha_bar":
        s = torch.linspace(0, 1, steps+1, device=device)
        w = 0.5 * (1 - torch.cos(torch.pi * s))
        idx = (T-1) * (1 - w)
    elif schedule == "karras":
        p = 2.0
        s = torch.linspace(0, 1, steps+1, device=device) ** p
        idx = (T-1) * (1 - s)
    else:
        raise ValueError("schedule inválido")

    ts = idx.round().clamp_(0, T-1).long()

    for i in range(steps):
        t  = ts[i].repeat(n).clone()
        tprev = ts[i+1].repeat(n).clone()
        x = diffusion.p_sample_step_ddim(
            model, x, t, tprev, eta=eta, clip_x0=clip_x0, noise=None)

    x = (x.clamp(-1,1) + 1)*0.5
    if save_path:
        save_image_grid(x, save_path, nrow=int(n**0.5))
    if backup is not None:
        model.load_state_dict(backup)
    model.train(was_training)
    return x


########### REST UTILS ################# 

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


