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