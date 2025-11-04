import torch

class EMA:
    """Exponential Moving Average para parámetros del modelo."""
    def __init__(self, model, decay=0.999, device=None):
        self.decay = decay
        self.shadow = []
        self.device = device
        for p in model.parameters():
            if p.requires_grad:
                self.shadow.append(p.detach().clone())
            else:
                self.shadow.append(None)

    @torch.no_grad()
    def update(self, model):
        i = 0
        for p in model.parameters():
            if p.requires_grad:
                if self.device is not None:
                    self.shadow[i] = self.shadow[i].to(self.device)
                self.shadow[i].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)
            i += 1

    @torch.no_grad()
    def copy_to(self, model):
        i = 0
        for p in model.parameters():
            if p.requires_grad:
                p.data.copy_(self.shadow[i].data)
            i += 1

    @torch.no_grad()
    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    @torch.no_grad()
    def load_state_dict(self, state):
        self.decay = state["decay"]

        self.shadow = state["shadow"]



@torch.no_grad()
def ema_health(ema, model, rel_tol: float = 5.0):
    """
    Devuelve (ok: bool, reason: str, rel_diff: float).
    ok=False si:
      - longitudes distintas
      - NaN/Inf en sombra
      - norma de sombra ~ 0
      - ||m - e|| / ||m|| > rel_tol
    """
    m_params = [p.detach().float().to("cpu") for p in model.parameters() if p.requires_grad]
    e_params = [s.detach().float().to("cpu") if s is not None else None for s in getattr(ema, "shadow", [])]

    if len(m_params) != len(e_params):
        return (False, "len_mismatch", float("inf"))

    m_flat, e_flat = [], []
    for mp, ep in zip(m_params, e_params):
        if ep is None:
            return (False, "shadow_none", float("inf"))
        m_flat.append(mp.view(-1))
        e_flat.append(ep.view(-1))
    m = torch.cat(m_flat, dim=0)
    e = torch.cat(e_flat, dim=0)

    if not torch.isfinite(e).all():
        return (False, "nan_or_inf_in_ema", float("inf"))

    m_norm = m.norm().item()
    e_norm = e.norm().item()
    if e_norm < 1e-12:
        return (False, "ema_zero_norm", float("inf"))
    if m_norm < 1e-12:
        return (False, "model_zero_norm", float("inf"))

    rel = (m - e).norm().item() / (m_norm + 1e-8)
    if rel > rel_tol:
        return (False, "large_rel_diff", rel)
    return (True, "ok", rel)

@torch.no_grad()
def ema_reinit_from_model(ema, model):
    """Copia 1:1 los pesos del modelo a la sombra de EMA (solo requires_grad)."""
    i = 0
    for p in model.parameters():
        if p.requires_grad:
            ema.shadow[i].data.copy_(p.data)
        i += 1

def ema_set_decay(ema, new_decay: float):
    try:
        ema.decay = float(new_decay)
    except Exception:
        pass
