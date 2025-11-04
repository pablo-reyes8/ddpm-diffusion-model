import os, time
import torch 
import torchvision.utils as vutils

from src.training_loops.train_one_epoch import *
from src.training_loops.training_utils import *

def _fmt_hms(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    h, m = divmod(m, 60)
    return f"{h:d}:{m:02d}:{s:02d}"

def _rule(w=92, ch="─"):  # línea separadora
    return ch * w
    
def _is_colab():
    return "google.colab" in sys.modules

def _ensure_drive_mounted():
    if _is_colab():
        drive_root = "/content/drive"
        if not os.path.isdir(drive_root):
            try:
                from google.colab import drive
                drive.mount(drive_root, force_remount=False)
            except Exception as e:
                print(f"[DRIVE] No se pudo montar automáticamente: {e}")


def _copy_ckpt_to_drive_fixed(src_path: str, drive_dir: str, fixed_name: str = "latest_ddpm.pt"):
    """Copia el checkpoint a Drive con nombre fijo; si existe, lo reemplaza."""
    try:
        if not drive_dir:
            return
        if drive_dir.startswith("/content/drive"):
            _ensure_drive_mounted()
        os.makedirs(drive_dir, exist_ok=True)
        dst_path = os.path.join(drive_dir, fixed_name)
        if os.path.exists(dst_path):
            os.remove(dst_path)
            print(f"└─ [DRIVE]  eliminado previo → {dst_path}")
        shutil.copy2(src_path, dst_path)
        print(f"└─ [DRIVE]  copiado (fixed) → {dst_path}")
    except Exception as e:
        print(f"└─ [DRIVE]  ERROR al copiar a Drive: {e}")


def train_ddpm(
    model, diffusion, train_loader, optimizer,
    ema, device="cuda",
    epochs=50, base_lr=2e-4, warmup_steps=1000,
    grad_clip=1.0, use_autocast=True, scaler=None,
    sample_every=5, sample_n=36, img_size=64,
    sample_fn=None,
    ckpt_dir="checkpoints", run_name="ddpm",
    save_every=5, save_last=True,
    resume_path=None, ckpt_utils=None,
    grad_accum_steps: int = 1,
    use_channels_last: bool = False,
    on_oom: str = "skip",
    # DIAG:
    log_every: int = 0,
    probe_timesteps: list[int] | None = None,
    log_mem: bool = False,
    log_grad_norm: bool = False,
    # Sampling diag:
    sample_seed: int | None = 1234,
    sample_steps: int | None = None,
    # control fino al reanudar
    reset_optimizer_state: bool = False,        # no cargar estado del optimizer
    override_lr: float | None = None,           # reasigna LR tras cargar
    override_weight_decay: float | None = None, # reasigna WD tras cargar
    override_ema_decay: float | None = None,    # reasigna EMA.decay tras cargar
    # Drive 
    drive_ckpt_dir: str | None = None,          # p.ej. "/content/drive/MyDrive/ddpm_checkpoints"
    copy_fixed_to_drive: bool = True,           # copiar con nombre fijo
    fixed_drive_name: str = "latest_ddpm.pt" ):

    os.makedirs(ckpt_dir, exist_ok=True)

    save_ckpt, load_ckpt = (None, None)
    if ckpt_utils is not None:
        save_ckpt, load_ckpt = ckpt_utils

    if scaler is None and use_autocast:
        scaler = make_grad_scaler(device=device, enabled=True)

    # Resumen
    global_step, start_epoch = 0, 0
    resumed = False
    if resume_path and load_ckpt is not None and os.path.exists(resume_path):
        opt_for_load = None if reset_optimizer_state else optimizer
        step_loaded, extra = load_ckpt(
            resume_path, model,
            optimizer=opt_for_load,
            scaler=scaler, ema=ema,
            map_location=device)

        if isinstance(extra, dict):
            global_step = int(extra.get("global_step", step_loaded or 0))
            start_epoch = int(extra.get("epoch", 0)) + 1
        print(f"[RESUME] Cargado: {resume_path} | global_step={global_step} | start_epoch={start_epoch}")

        # Overrides SOLO si reanudamos
        if reset_optimizer_state:
            print("[RESUME] Optimizer: estado NO cargado (reset).")
        if override_lr is not None:
            for g in optimizer.param_groups:
                g["lr"] = float(override_lr)
            print(f"[RESUME] override_lr → {override_lr:.3e}")
        if override_weight_decay is not None:
            for g in optimizer.param_groups:
                g["weight_decay"] = float(override_weight_decay)
            print(f"[RESUME] override_weight_decay → {override_weight_decay:.3e}")
        if override_ema_decay is not None:
            # algunas clases EMA guardan decay en state; aquí lo forzamos
            if hasattr(ema, "decay"):
                ema.decay = float(override_ema_decay)
            print(f"[RESUME] override_ema_decay → {override_ema_decay:.6f}")
        resumed = True

    #Header
    ema_decay = getattr(ema, "decay", None)
    ema_str   = f"{ema_decay:.6f}" if isinstance(ema_decay, (float, int)) else "on"
    print(_rule())
    print(f"DDPM run: {run_name}")
    print(f"Device: {device} | autocast: {use_autocast} | EMA: {ema_str} | "
          f"epochs: {epochs} | base_lr: {base_lr:.2e} | warmup_steps: {warmup_steps}")
    if resumed:
        print("Overrides activos al reanudar:",
              f"reset_opt={reset_optimizer_state}",
              f"override_lr={override_lr}",
              f"override_wd={override_weight_decay}",
              f"override_ema={override_ema_decay}", sep=" ")
    print(_rule())
    print(f"{'ep':>3} | {'step':>8} | {'loss':>10} | {'lr':>9} | "
          f"{'batches':>8} | {'images':>8} | {'imgs/s':>7} | {'time':>8} | {'warmup':>6}")
    print(_rule())

    total_time = 0.0

    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        avg_loss, n_batches, n_images, global_step = train_one_epoch(
            model=model, diffusion=diffusion, dataloader=train_loader, optimizer=optimizer,
            scaler=scaler, ema=ema, device=device,
            grad_clip=grad_clip, use_autocast=use_autocast,
            grad_accum_steps=grad_accum_steps, use_channels_last=use_channels_last, on_oom=on_oom,
            base_lr=base_lr, warmup_steps=warmup_steps, global_step=global_step,
            log_every=log_every, probe_timesteps=probe_timesteps,
            log_mem=log_mem, log_grad_norm=log_grad_norm)

        sec = time.time() - t0
        total_time += sec
        ips = (n_images / sec) if sec > 0 else 0.0
        lr_now = optimizer.param_groups[0]["lr"]
        warm_prog = 0.0 if not warmup_steps else min(1.0, global_step / float(warmup_steps))

        print(f"{epoch:3d} | {global_step:8d} | {avg_loss:10.5f} | {lr_now:9.2e} | "
              f"{n_batches:8d} | {n_images:8d} | {ips:7.1f} | {_fmt_hms(sec):>8} | {int(100*warm_prog):3d}%")

        # Muestras (EMA swap temporal)
        if (sample_fn is not None) and ((epoch % sample_every == 0) or (epoch == epochs - 1)):
            out_path = os.path.join(ckpt_dir, f"{run_name}_samples_e{epoch:03d}.png")
            backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
            ema.copy_to(model)
            if sample_seed is not None:
                torch.manual_seed(sample_seed)
            _ = sample_fn(model, diffusion, n=sample_n, img_size=img_size,
                          device=device, save_path=out_path)
            model.load_state_dict(backup)
            print(f"└─ [SAMPLE] grid → {out_path}")

        # Checkpoints (local + copia a Drive con nombre fijo)
        if (save_ckpt is not None) and ((epoch % save_every == 0) or (epoch == epochs - 1)):
            ckpt_path = os.path.join(ckpt_dir, f"{run_name}_e{epoch:03d}.pt")
            save_ckpt(ckpt_path, model, optimizer, scaler, ema,
                      step=global_step, extra={"epoch": epoch, "global_step": global_step})
            print(f"└─ [CKPT]   saved → {ckpt_path}")

            # copia fija a Drive 
            if copy_fixed_to_drive and drive_ckpt_dir:
                _copy_ckpt_to_drive_fixed(ckpt_path, drive_ckpt_dir, fixed_name=fixed_drive_name)

    if save_last and (save_ckpt is not None):
        ckpt_path = os.path.join(ckpt_dir, f"{run_name}_last.pt")
        save_ckpt(ckpt_path, model, optimizer, scaler, ema,
                  step=global_step, extra={"epoch": epochs-1, "global_step": global_step})
        print(f"└─ [CKPT]   saved → {ckpt_path}")

        # copia fija a Drive (last)
        if copy_fixed_to_drive and drive_ckpt_dir:
            _copy_ckpt_to_drive_fixed(ckpt_path, drive_ckpt_dir, fixed_name=fixed_drive_name)

    print(_rule())
    print(f"Entrenamiento finalizado en {_fmt_hms(total_time)}")
    print(_rule())




