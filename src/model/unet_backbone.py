import math
from typing import Tuple, Sequence, Set, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.attention import * 


class ResBlock(nn.Module):
    """
    Residual block con GN + SiLU + Conv, condicionado por el embedding temporal.
    Cada bloque puede cambiar el # de canales (in_ch -> out_ch).
    """
    def __init__(self, in_ch: int, out_ch: int, time_dim: int, dropout: float = 0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch

        self.norm1 = group_norm(in_ch)
        self.act1  = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)

        # Proyección del tiempo al sesgo por canal
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_ch))

        self.norm2 = group_norm(out_ch)
        self.act2  = nn.SiLU()
        self.drop  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)

        # Skip si cambia número de canales
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act1(self.norm1(x)))
        # Añadimos conditioning del tiempo como sesgo por canal
        # (B, out_ch) -> (B, out_ch, 1, 1)
        t_bias = self.time_proj(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = h + t_bias
        h = self.conv2(self.drop(self.act2(self.norm2(h))))
        return h + self.skip(x)
    

class Downsample(nn.Module):
    """Downsample por factor 2 con conv (stride=2) para evitar aliasing excesivo."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    """Upsample ×2 por interpolación + conv (evita checkerboard)."""
    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)
    


class UNetDenoiser(nn.Module):
    """
    U-Net para DDPM (predice ε).
    - in_channels: 3 (RGB)
    - base_channels: canales iniciales (p.ej., 128)
    - channel_mults: multiplicadores por nivel (64->32->16->8->...); p.ej., (1,2,2,2) para 64x64
    - num_res_blocks: # de ResBlocks por resolución en encoder/decoder
    - attn_resolutions: resoluciones (H o W) donde aplicar atención (p.ej., {16, 8})
    - time_dim: dimensión del embedding temporal proyectado
    """
    def __init__(self,
        in_channels: int = 3,
        base_channels: int = 128,
        channel_mults: Sequence[int] = (1, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: Set[int] = frozenset({16, 8}),
        time_embed_dim: int = 512,
        dropout: float = 0.0,
        num_heads: int = 4,
        head_dim: int = 64,
        img_resolution: int = 64):

        super().__init__()

        # Embedding temporal
        self.time_pos_emb = SinusoidalPosEmb(time_embed_dim)
        self.time_mlp  = TimeMLP(time_embed_dim, time_embed_dim)

        # Conv de entrada/salida
        self.in_conv  = nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1)
        self.out_norm = group_norm(base_channels)
        self.out_act  = nn.SiLU()
        self.out_conv = nn.Conv2d(base_channels, in_channels, kernel_size=3, padding=1)

        # Construcción encoder
        ch = base_channels
        self.downs = nn.ModuleList()
        self.skip_shapes = []

        in_ch = ch
        resolutions = [img_resolution]
        for mult in channel_mults:
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, time_dim=time_embed_dim, dropout=dropout)) # Bloques ResNet
                in_ch = out_ch
                # atención opcional en esta resolución
                if resolutions[-1] in attn_resolutions:
                    blocks.append(AttnBlock(in_ch, num_heads=num_heads, head_dim=head_dim)) # Atencion en bajas capas

            # Al final del nivel, si no es último, downsample
            down = nn.Module()
            down.blocks = blocks
            down.down = Downsample(in_ch) if mult != channel_mults[-1] else nn.Identity() # Reducimos resolucion en Encoder
            self.downs.append(down)
            if mult != channel_mults[-1]:
                resolutions.append(resolutions[-1] // 2)

        # Bottleneck
        self.mid = nn.ModuleList([
            ResBlock(in_ch, in_ch, time_dim=time_embed_dim, dropout=dropout),
            AttnBlock(in_ch, num_heads=num_heads, head_dim=head_dim) if (resolutions[-1]//2) in attn_resolutions else nn.Identity(),
            ResBlock(in_ch, in_ch, time_dim=time_embed_dim, dropout=dropout),])

        # Construcción decoder (mirror)
        self.ups = nn.ModuleList()
        for mult in reversed(channel_mults):
            out_ch = base_channels * mult
            blocks = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                blocks.append(ResBlock(in_ch + out_ch if _ == 0 else in_ch, out_ch, time_dim=time_embed_dim, dropout=dropout))
                in_ch = out_ch
                res_for_attn = resolutions[-1] if mult == channel_mults[0] else (resolutions[-1] * 2)

                if res_for_attn in attn_resolutions:
                    blocks.append(AttnBlock(in_ch, num_heads=num_heads, head_dim=head_dim))

            up = nn.Module()
            up.blocks = blocks
            up.up = Upsample(in_ch) if mult != channel_mults[0] else nn.Identity() # Aumentamos resolucion en Decoder
            self.ups.append(up)

            if mult != channel_mults[0]:
                resolutions.append(resolutions[-1] * 2)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
      """
      x: (B, 3, H, W) en [-1, 1]
      t: (B,) timesteps (int o float)
      retorna: eps_pred (B, 3, H, W)
      """
      # Embedding temporal
      t_emb = self.time_mlp(self.time_pos_emb(t))  # (B, time_dim)

      # Encoder
      h = self.in_conv(x)
      skips = []
      cur = h
      for down in self.downs:
          for blk in down.blocks:
              if isinstance(blk, ResBlock):
                  cur = blk(cur, t_emb)
              else:
                  cur = blk(cur)
          skips.append(cur)
          cur = down.down(cur)

      # Bottleneck
      for blk in self.mid:
          if isinstance(blk, ResBlock):
              cur = blk(cur, t_emb)
          else:
              cur = blk(cur)

      # Decoder
      for up in self.ups:
          # subimos resolución si este nivel lo requiere
          if not isinstance(up.up, torch.nn.Identity):
              cur = up.up(cur)

          skip = skips.pop()
          if cur.shape[-2:] != skip.shape[-2:]:
              cur = torch.nn.functional.interpolate(cur, size=skip.shape[-2:], mode="nearest")

          # concatenamos canales
          cur = torch.cat([cur, skip], dim=1)

          for blk in up.blocks:
              if isinstance(blk, ResBlock):
                  cur = blk(cur, t_emb)
              else:
                  cur = blk(cur)

      # Salida
      out = self.out_conv(self.out_act(self.out_norm(cur)))
      return out  # ε̂: (B, 3, H, W)


def build_unet_64x64(
    in_channels: int = 3,
    base_channels: int = 128,
    channel_mults: Tuple[int, ...] = (1, 2, 2, 2),
    num_res_blocks: int = 2,
    attn_resolutions: Set[int] = frozenset({16, 8}),
    time_embed_dim: int = 512,
    dropout: float = 0.1,
    num_heads: int = 4,
    head_dim: int = 64,):

    return UNetDenoiser(
        in_channels=in_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        attn_resolutions=attn_resolutions,
        time_embed_dim=time_embed_dim,
        dropout=dropout,
        num_heads=num_heads,
        head_dim=head_dim,
        img_resolution=64)
