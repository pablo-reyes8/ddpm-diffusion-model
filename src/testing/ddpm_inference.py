import torchvision.utils as vutils
import torch
import os, math

@torch.no_grad()
def ddpm_infer_sample(
    model,
    diffusion,
    n: int = 36,
    img_size: int = 64,
    device: str = "cuda",*,
    ema=None,
    out_path: str = "samples_ddpm.png",
    save_individual: bool = False,
    out_dir: str = "samples_individual",
    seed: int | None = 1234):

    """
    Genera n imágenes con DDPM ancestral (T pasos) y guarda un grid.
    Requiere: diffusion.p_sample_step(model_eps, x_t, t)
    """
    model_was_training = model.training
    model.eval()

    backup_state = None
    if ema is not None:
        backup_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)

    if seed is not None:
        torch.manual_seed(seed)

    B = n
    x = torch.randn(B, 3, img_size, img_size, device=device)

    for i in reversed(range(diffusion.T)):
        t = torch.full((B,), i, device=device, dtype=torch.long)
        x = diffusion.p_sample_step(lambda x_t, t_t: model(x_t, t_t), x, t)

    x = (x.clamp(-1, 1) + 1) * 0.5

    nrow = int(math.sqrt(B))
    grid = vutils.make_grid(x, nrow=nrow, padding=2)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vutils.save_image(grid, out_path)
    print(f"[INFER] Grid guardado en: {out_path}")

    if save_individual:
        os.makedirs(out_dir, exist_ok=True)
        for idx in range(B):
            vutils.save_image(x[idx], os.path.join(out_dir, f"img_{idx:03d}.png"))
        print(f"[INFER] {B} imágenes individuales guardadas en: {out_dir}")

    if backup_state is not None:
        model.load_state_dict(backup_state)
    model.train(model_was_training)

    return grid


@torch.no_grad()
def render_denoise_strip(
    model,
    diffusion,
    *,
    img_size: int = 64,
    device: str = "cuda",
    ema=None,                   # opcional: usar EMA para muestrear
    seed: int | None = 1234,
    out_path: str = "denoise_strip.png",
    capture_steps: list[int] | None = None,   # índices t del reverse a capturar
    pad: int = 2  ):
    
    """
    Genera una única muestra x_T~N(0,I) y la denoisa T→0.
    Captura snapshots en 'capture_steps' y guarda un grid 1×K (una fila).

    - 'capture_steps' son índices de t (0..T-1) del reverse; si None, toma ~20 equiespaciados.
    - Devuelve el tensor (C,H,W) del grid (normalizado en [0,1]).
    """
    was_training = model.training
    model.eval()

    # swap temporal a EMA si se provee
    backup = None
    if ema is not None:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)

    if seed is not None:
        torch.manual_seed(seed)

    T = diffusion.T
    if capture_steps is None:
        K = 20
        capture_steps = [int(x) for x in torch.linspace(T-1, 0, K).round().tolist()]

    capture_set = set(capture_steps)

    x = torch.randn(1, 3, img_size, img_size, device=device)
    frames = []

    for t_int in range(T-1, -1, -1):
        t = torch.full((1,), t_int, device=device, dtype=torch.long)
        x = diffusion.p_sample_step(lambda x_t, tt: model(x_t, tt), x, t)
        if t_int in capture_set:
            x_vis = (x.clamp(-1, 1) + 1) * 0.5  # [0,1]
            frames.append(x_vis[0].detach().cpu())

    grid = vutils.make_grid(torch.stack(frames, 0), nrow=len(frames), padding=pad)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vutils.save_image(grid, out_path)
    print(f"[DENOISE] strip 1×{len(frames)} guardado → {out_path}")

    if backup is not None:
        model.load_state_dict(backup)
    model.train(was_training)
    return grid

