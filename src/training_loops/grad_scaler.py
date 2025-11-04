import inspect
from contextlib import contextmanager, nullcontext
import torch 

def make_grad_scaler(device: str = "cuda", enabled: bool = True):
    """
    Devuelve un GradScaler compatible con tu versión de PyTorch.
    - Si AMP no está habilitado, devuelve None.
    - Soporta torch.amp.GradScaler('cuda'|'cpu') (algunas versiones)
      y torch.cuda.amp.GradScaler() (otras versiones).
    """
    if not enabled:
        return None

    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            sig = inspect.signature(torch.amp.GradScaler)
            if len(sig.parameters) >= 1:
                return torch.amp.GradScaler(device if device in ("cuda", "cpu") else "cuda")
            else:
                return torch.amp.GradScaler()
        except Exception:
            pass

    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler()

    return None

@contextmanager
def autocast_ctx(device: str = "cuda", enabled: bool = True):
    """
    Contexto autocast compatible:
      - Usa torch.amp.autocast(device_type='cuda'/'cpu') si existe.
      - Si no, usa torch.cuda.amp.autocast() cuando device='cuda'.
      - Si AMP off, nullcontext().
    """
    if not enabled:
        with nullcontext():
            yield
        return

    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        with torch.amp.autocast(device_type=("cuda" if device == "cuda" else "cpu")):
            yield
        return

    if device == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        with torch.cuda.amp.autocast():
            yield
        return

    with nullcontext():
        yield

