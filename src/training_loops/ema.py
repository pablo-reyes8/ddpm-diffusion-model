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
      - longitudes distintas (solo si hay params congelados)
      - NaN/Inf en sombra
      - norma de sombra ~ 0
      - ||m - e|| / ||m|| > rel_tol
    """
    # Aplanado robusto (maneja no-contiguos)
    def _flat(t):
        return t.detach().float().cpu().reshape(-1)

    # Solo params con grad (EMA se construyó así)
    m_params = [p for p in model.parameters() if p.requires_grad]
    e_params = [s for s in getattr(ema, "shadow", []) if s is not None]

    # Si por alguna razón hay mismatch de conteo, marca inválida y deja que se repare.
    if len(m_params) != len(e_params):
        return (False, "len_mismatch", float("inf"))

    m_flat = torch.cat([_flat(p) for p in m_params], dim=0)
    e_flat = torch.cat([_flat(s) for s in e_params], dim=0)

    if not torch.isfinite(e_flat).all():
        return (False, "nan_or_inf_in_ema", float("inf"))

    m_norm = m_flat.norm().item()
    e_norm = e_flat.norm().item()
    if e_norm < 1e-12:
        return (False, "ema_zero_norm", float("inf"))
    if m_norm < 1e-12:
        return (False, "model_zero_norm", float("inf"))

    rel = (m_flat - e_flat).norm().item() / (m_norm + 1e-8)
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

