from torchvision.utils import make_grid, save_image
import torch 
import time

from src.training_loops.chekpoints import * 
from src.training_loops.ema import * 
from src.training_loops.grad_scaler import * 
from src.training_loops.training_utils import *



def train_one_epoch(
    model,
    diffusion,
    dataloader,
    optimizer,
    *,
    scaler=None,
    ema=None,
    device: str = "cuda",
    max_batches: int | None = None,
    grad_clip: float | None = 1.0,
    use_autocast: bool = True,
    grad_accum_steps: int = 1,
    use_channels_last: bool = False,
    on_oom: str = "skip",
    # warmup por paso
    base_lr: float | None = None,
    warmup_steps: int | None = None,
    global_step: int = 0,
    # DIAGNÓSTICOS
    log_every: int = 0,                       # imprime cada N steps (0 = off)
    probe_timesteps: list[int] | None = None, # e.g., [50, 200, 500, 800]
    log_mem: bool = False,                    # imprime memoria GPU
    log_grad_norm: bool = False,              # imprime ||g||
):
    model.train()
    if use_channels_last:
        model.to(memory_format=torch.channels_last)

    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    total_loss, n_seen_batches, n_seen_images = 0.0, 0, 0

    # encabezado de sección "In-epoch" y baseline
    did_print_in_epoch_header = False
    if log_every and global_step == 0:
        with torch.no_grad():
            xb = torch.randn(32, 3, diffusion.img_size, diffusion.img_size, device=device)
            base = float((xb**2).mean().item())
        print("┆ In-epoch statistics")
        print("┆   (baseline)  ε-MSE ≈ {:.3f}  (esperado ~1.0)".format(base))
        print("┆   {:>8} | {:>9} | {:>8} | {:>8} | {:>10}{}".format(
            "step", "lr", "loss", "dt(ms)", "grad_norm",
            (" | probes[t]" if probe_timesteps else "")
        ))
        print("┆   " + "─"*72)
        did_print_in_epoch_header = True

    ### Begin Training ###
    for i, (x, _) in enumerate(dataloader):
        if (max_batches is not None) and (i >= max_batches):
            break
        try:
            t_start = time.perf_counter()

            x = x.to(device, non_blocking=True)
            if use_channels_last:
                x = x.to(memory_format=torch.channels_last)

            B = x.size(0)
            t = diffusion.sample_timesteps(B, device=device)

            with autocast_ctx(device=device, enabled=bool(use_autocast)):
                loss = diffusion.loss_simple(model, x, t) / grad_accum_steps

            if use_autocast and (scaler is not None):
                scaler.scale(loss).backward()
            else:
                loss.backward()

            step_now = ((i + 1) % grad_accum_steps) == 0
            gnorm = None
            if step_now:
                # Warmup por paso
                if (base_lr is not None) and (warmup_steps is not None) and (warmup_steps > 0):
                    lr = base_lr * min(1.0, (global_step + 1) / warmup_steps)
                    for g in optimizer.param_groups:
                        g["lr"] = lr

                did_unscale = False

                # Grad norm (antes de clip) si se pide
                if log_grad_norm:
                    if use_autocast and (scaler is not None) and (not did_unscale):
                        scaler.unscale_(optimizer)
                        did_unscale = True
                    gnorm = compute_grad_norm(model)

                # Clip + step
                if grad_clip is not None:
                    if use_autocast and (scaler is not None) and (not did_unscale):
                        scaler.unscale_(optimizer)
                        did_unscale = True
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                if use_autocast and (scaler is not None):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                if ema is not None:
                    ema.update(model)

                global_step += 1

            total_loss += float(loss.detach()) * grad_accum_steps
            n_seen_batches += 1
            n_seen_images  += B

            # logs compactos por step ─
            if log_every and (global_step % log_every == 0) and step_now:
                if not did_print_in_epoch_header:
                    print("┆ In-epoch statistics")
                    print("┆   {:>8} | {:>9} | {:>8} | {:>8} | {:>10}{}".format(
                        "step", "lr", "loss", "dt(ms)", "grad_norm",
                        (" | probes[t]" if probe_timesteps else "")
                    ))
                    print("┆   " + "─"*72)
                    did_print_in_epoch_header = True

                probe_msg = ""
                if probe_timesteps:
                    with torch.no_grad(), autocast_ctx(device=device, enabled=False):
                        vals = []
                        for tau in probe_timesteps:
                            t_fix = torch.full((B,), int(tau), device=device, dtype=torch.long)
                            v = diffusion.loss_simple(model, x, t_fix).item()
                            vals.append(f"t={tau}:{v:.3f}")
                        probe_msg = " | " + " ".join(vals)

                # (opcional) memoria, pero sin romper el layout
                mem_msg = ""
                if log_mem:
                    alloc, reserv = gpu_mem_mb(device)
                    mem_msg = f" | mem={alloc:.0f}/{reserv:.0f}MB"

                lr_now = optimizer.param_groups[0]["lr"]
                dt = (time.perf_counter() - t_start) * 1000.0
                gn_str = (f"{gnorm:.2e}" if (gnorm is not None) else "—")
                loss_val = (loss.detach() * grad_accum_steps).item()

                print("┆   {:8d} | {:9.2e} | {:8.4f} | {:8.1f} | {:>10}{}{}".format(global_step, lr_now, loss_val, dt, gn_str, mem_msg, probe_msg))

        except RuntimeError as e:
            if ("CUDA out of memory" in str(e)) and (on_oom == "skip"):
                import gc
                gc.collect(); torch.cuda.empty_cache()
                print(f"[WARN][OOM] Batch {i} omitido. Limpié cache y sigo…")
                optimizer.zero_grad(set_to_none=True)
                continue
            else:
                raise

    avg_loss = total_loss / max(1, n_seen_batches)
    return avg_loss, n_seen_batches, n_seen_images, global_step