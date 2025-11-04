import torchvision.utils as vutils
import torch
import os, math


@torch.no_grad()
def ddim_infer_sample(
    model,
    diffusion,
    n: int = 36,
    img_size: int = 64,
    device: str = "cuda",
    *,
    ema=None,
    out_path: str = "samples_ddim.png",
    save_individual: bool = False,
    out_dir: str = "samples_individual",
    seed: int | None = 1234,
    steps: int = 50,
    eta: float = 0.0,
    schedule_kind: str = "t_linear",   # "t_linear" o "alpha_bar_cosine"
    schedule_idx: list[int] | None = None):
  
    was_training = model.training
    model.eval()

    backup_state = None
    if ema is not None:
        backup_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)

    if seed is not None:
        torch.manual_seed(seed)

    B = n
    x = torch.randn(B, 3, img_size, img_size, device=device)

    T = diffusion.T

    def build_schedule():
        if schedule_idx is not None:
            s = torch.tensor(sorted({int(t) for t in schedule_idx}, reverse=True), device=device)
            if s[-1].item() != 0:
                s = torch.cat([s, torch.zeros(1, device=device, dtype=s.dtype)])
            return s

        if schedule_kind == "t_linear":
            # pasos equiespaciados en t, incluyendo 0
            s = torch.linspace(T-1, 0, steps, device=device)
            s = torch.unique_consecutive(s.round().long(), dim=0)
            if s[-1].item() != 0:
                s = torch.cat([s, torch.zeros(1, device=device, dtype=s.dtype)])
            return s

        elif schedule_kind == "alpha_bar_cosine":
            a_bar = diffusion.alphas_cumprod 
            u = torch.linspace(0.0, 1.0, steps, device=device)
            targets = (1.0 - u)  
            s_list = []
            for z in targets:
                # índice t tal que ᾱ_t ≈ z (como a_bar es monótona ↓, usamos argmin |ᾱ - z|)
                t_idx = (a_bar - z).abs().argmin()
                s_list.append(int(t_idx.item()))
            s = torch.tensor(sorted(set(s_list), reverse=True), device=device)
            if s[-1].item() != 0:
                s = torch.cat([s, torch.zeros(1, device=device, dtype=s.dtype)])
            return s

        else:
            raise ValueError(f"schedule_kind desconocido: {schedule_kind}")

    schedule = build_schedule()
    #  loop DDIM: t -> t_prev -
    for i in range(len(schedule) - 1):
        t_cur  = schedule[i]
        t_prev = schedule[i+1]
        t_cur_t  = torch.full((B,), int(t_cur.item()),  device=device, dtype=torch.long)
        t_prev_t = torch.full((B,), int(t_prev.item()), device=device, dtype=torch.long)

        x = diffusion.p_sample_step_ddim(
            lambda x_t, tt: model(x_t, tt),
            x_t=x,
            t=t_cur_t,
            t_prev=t_prev_t,
            eta=eta,
            clip_x0=True,
            noise=None,)

    x_vis = (x.clamp(-1, 1) + 1) * 0.5
    nrow = int(math.sqrt(B)) if int(math.sqrt(B))**2 == B else math.ceil(math.sqrt(B))
    grid = vutils.make_grid(x_vis, nrow=nrow, padding=2)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vutils.save_image(grid, out_path)
    print(f"[INFER-DDIM] Grid → {out_path}  (steps={len(schedule)-1}, eta={eta}, schedule={schedule_kind})")

    if save_individual:
        os.makedirs(out_dir, exist_ok=True)
        for i in range(B):
            vutils.save_image(x_vis[i], os.path.join(out_dir, f"img_{i:03d}.png"))

    if backup_state is not None:
        model.load_state_dict(backup_state)
    model.train(was_training)
    return grid


@torch.no_grad()
def render_denoise_strip_ddim(
    model,
    diffusion,
    *,
    img_size: int = 64,
    device: str = "cuda",
    ema=None,                         # opcional: usar EMA para muestrear
    seed = 1234,
    out_path: str = "denoise_strip_ddim.png",
    capture_steps = None,   # t en espacio original (0..T-1)
    pad: int = 2,
    # DDIM params
    steps: int = 50,                  # nº de pasos DDIM (<< T)
    eta: float = 0.0,                 # 0 = determinista
    schedule_kind: str = "linear",    # "linear" o "cosine"
    schedule_idx = None,):
    """
    Genera UNA muestra con DDIM (steps<<T) y guarda un grid horizontal (1×K) con
    snapshots de la trayectoria reverse. Si 'capture_steps' es None, se toman
    ~min(20, steps) puntos del schedule DDIM.

    Requiere que diffusion tenga: .T, .alphas_cumprod y .p_sample_step_ddim(...)
    """
    was_training = model.training
    model.eval()

    # Swap temporal a EMA si aplica
    backup = None
    if ema is not None:
        backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema.copy_to(model)

    if seed is not None:
        torch.manual_seed(seed)

    T = diffusion.T

    #Construir schedule DDIM (índices t descendentes)
    device = torch.device(device)
    if schedule_idx is None:
        if schedule_kind == "cosine":
            s = torch.linspace(0, 1, steps, device=device)
            w = 0.5 * (1 - torch.cos(math.pi * s))  # 0..1
            idx = torch.round((T - 1) * (1 - w)).long()
        else:
            idx = torch.round(torch.linspace(T - 1, 0, steps, device=device)).long()
        ddim_schedule = sorted(set(idx.tolist()), reverse=True)
    else:
        ddim_schedule = sorted(set(int(t) for t in schedule_idx), reverse=True)

    if capture_steps is None:
        K = min(17, len(ddim_schedule))
        pick = torch.linspace(0, len(ddim_schedule) - 1, K).round().long().tolist()
        capture_steps = [ddim_schedule[i] for i in pick]
    capture_set = set(int(t) for t in capture_steps)

    #  Muestreo DDIM de una sola muestra
    x = torch.randn(1, 3, img_size, img_size, device=device)
    frames = []

    for i, t_cur in enumerate(ddim_schedule):
        t_cur_t  = torch.full((1,), t_cur, device=device, dtype=torch.long)
        t_prev   = ddim_schedule[i + 1] if i + 1 < len(ddim_schedule) else 0
        t_prev_t = torch.full((1,), t_prev, device=device, dtype=torch.long)

        x = diffusion.p_sample_step_ddim(
            lambda xt, tt: model(xt, tt),
            x_t=x,
            t=t_cur_t,
            t_prev=t_prev_t,
            eta=eta,
            clip_x0=True,
            noise=None)

        if t_cur in capture_set:
            x_vis = (x.clamp(-1, 1) + 1) * 0.5
            frames.append(x_vis[0].detach().cpu())

    grid = vutils.make_grid(torch.stack(frames, 0), nrow=len(frames), padding=pad)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    vutils.save_image(grid, out_path)

    print(f"[DENOISE-DDIM] strip 1×{len(frames)} guardado → {out_path} "
          f"(steps={len(ddim_schedule)}, eta={eta})")

    if backup is not None:
        model.load_state_dict(backup)

    model.train(was_training)
    return grid

