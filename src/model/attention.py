import math
from typing import Tuple, Sequence, Set, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    """Embedding sinusoidal estándar para timesteps (t) -> R^dim."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        device = t.device
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = t.float().unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([emb.sin(), emb.cos()], dim=1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb


class TimeMLP(nn.Module):
    """Proyecta el embedding sinusoidal a un espacio (time_dim) con activación."""
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.SiLU(),
            nn.Linear(out_dim, out_dim),)

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)  # (B, out_dim)
    

def group_norm(channels: int, num_groups: int = 32) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(num_groups, channels), num_channels=channels, eps=1e-6, affine=True)


class AttnBlock(nn.Module):
    """
    Auto-atención espacial multi-head simple para mapas (B, C, H, W).
    Pre-norm → QKV → atención → proyección.
    Usar en resoluciones bajas (p.ej., 16x16 y 8x8).
    """
    def __init__(self, channels: int, num_heads: int = 4, head_dim: int = 64):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner = num_heads * head_dim

        self.norm = group_norm(channels)
        self.qkv  = nn.Conv2d(channels, inner * 3, kernel_size=1, bias=False) # Proyeccion KQV, Kernel de 1 es una proyeccion
        self.proj = nn.Conv2d(inner, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_in = self.norm(x)

        qkv = self.qkv(h_in)  # (B, 3*inner, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)  # Dividimos el tensor proyectado en (B, inner, H, W) cada uno

        # reshapes a (B, heads, head_dim, HW)
        def reshape_heads(t):
            t = t.reshape(b, self.num_heads, self.head_dim, h * w)
            return t

        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # atenciones: (B, heads, HW, HW)
        attn = torch.einsum('bhcn,bhdn->bhcd', q, k) / math.sqrt(self.head_dim) # Einstein summation
        attn = attn.softmax(dim=-1)

        out = torch.einsum('bhcd,bhdn->bhcn', attn, v)
        out = out.reshape(b, self.num_heads * self.head_dim, h, w)
        out = self.proj(out)
        return x + out