"""
Sanity checks for U-Net backbone components.
Tests ResBlocks, Upsample/Downsample, and full U-Net architecture.
"""
import torch
import torch.nn as nn
from src.model.unet_backbone import (
    UNetDenoiser, 
    ResBlock, 
    Upsample, 
    Downsample,
    build_unet_64x64
)
from src.model.attention import SinusoidalPosEmb, TimeMLP, group_norm


def test_res_block():
    """Test ResBlock forward pass and shape consistency."""
    print("[TEST] ResBlock...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, C_in, C_out, H, W = 4, 64, 128, 32, 32
    time_dim = 512
    
    block = ResBlock(C_in, C_out, time_dim, dropout=0.1).to(device)
    x = torch.randn(B, C_in, H, W, device=device)
    t_emb = torch.randn(B, time_dim, device=device)
    
    with torch.no_grad():
        out = block(x, t_emb)
    
    assert out.shape == (B, C_out, H, W), f"ResBlock output shape mismatch: {out.shape}"
    assert torch.isfinite(out).all(), "ResBlock output contains NaN/Inf"
    print(f"  [OK] ResBlock: {x.shape} -> {out.shape}")
    
    # Test with same channels (skip connection)
    block_same = ResBlock(C_out, C_out, time_dim, dropout=0.0).to(device)
    with torch.no_grad():
        out2 = block_same(out, t_emb)
    assert out2.shape == out.shape, "ResBlock with same channels failed"
    print(f"  [OK] ResBlock (same channels): {out.shape} -> {out2.shape}")


def test_upsample_downsample():
    """Test Upsample and Downsample modules."""
    print("[TEST] Upsample/Downsample...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, C, H, W = 4, 128, 32, 32
    
    # Downsample
    downsample = Downsample(C).to(device)
    x = torch.randn(B, C, H, W, device=device)
    with torch.no_grad():
        x_down = downsample(x)
    assert x_down.shape == (B, C, H//2, W//2), f"Downsample shape mismatch: {x_down.shape}"
    assert torch.isfinite(x_down).all(), "Downsample output contains NaN/Inf"
    print(f"  [OK] Downsample: {x.shape} -> {x_down.shape}")
    
    # Upsample
    upsample = Upsample(C).to(device)
    with torch.no_grad():
        x_up = upsample(x_down)
    assert x_up.shape == (B, C, H, W), f"Upsample shape mismatch: {x_up.shape}"
    assert torch.isfinite(x_up).all(), "Upsample output contains NaN/Inf"
    print(f"  [OK] Upsample: {x_down.shape} -> {x_up.shape}")


def test_time_embeddings():
    """Test time embedding components."""
    print("[TEST] Time Embeddings...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B = 8
    time_dim = 512
    
    pos_emb = SinusoidalPosEmb(time_dim).to(device)
    time_mlp = TimeMLP(time_dim, time_dim).to(device)
    
    t = torch.randint(0, 1000, (B,), device=device)
    
    with torch.no_grad():
        t_emb = pos_emb(t)
        assert t_emb.shape == (B, time_dim), f"Positional embedding shape: {t_emb.shape}"
        assert torch.isfinite(t_emb).all(), "Positional embedding contains NaN/Inf"
        
        t_proj = time_mlp(t_emb)
        assert t_proj.shape == (B, time_dim), f"Time MLP output shape: {t_proj.shape}"
        assert torch.isfinite(t_proj).all(), "Time MLP output contains NaN/Inf"
    
    print(f"  [OK] SinusoidalPosEmb: {t.shape} -> {t_emb.shape}")
    print(f"  [OK] TimeMLP: {t_emb.shape} -> {t_proj.shape}")


def test_unet_denoiser():
    """Test full U-Net denoiser architecture."""
    print("[TEST] UNetDenoiser...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    B, C, H, W = 4, 3, 64, 64
    
    # Test without attention
    model = build_unet_64x64(
        in_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions=set(),  # No attention
        dropout=0.1
    ).to(device)
    
    x = torch.randn(B, C, H, W, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    
    with torch.no_grad():
        eps_pred = model(x, t)
    
    assert eps_pred.shape == x.shape, f"U-Net output shape mismatch: {eps_pred.shape}"
    assert torch.isfinite(eps_pred).all(), "U-Net output contains NaN/Inf"
    
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  [OK] UNetDenoiser (no attention): {x.shape} -> {eps_pred.shape}")
    print(f"  [OK] Parameters: {param_count:,}")
    
    # Test with attention
    model_attn = build_unet_64x64(
        in_channels=3,
        base_channels=128,
        channel_mults=(1, 2, 2, 2),
        num_res_blocks=2,
        attn_resolutions={16, 8},
        dropout=0.1
    ).to(device)
    
    with torch.no_grad():
        eps_pred_attn = model_attn(x, t)
    
    assert eps_pred_attn.shape == x.shape, "U-Net with attention output shape mismatch"
    assert torch.isfinite(eps_pred_attn).all(), "U-Net with attention contains NaN/Inf"
    
    param_count_attn = sum(p.numel() for p in model_attn.parameters())
    print(f"  [OK] UNetDenoiser (with attention): {x.shape} -> {eps_pred_attn.shape}")
    print(f"  [OK] Parameters: {param_count_attn:,}")
    
    # Attention should have more parameters
    assert param_count_attn > param_count, "Attention model should have more parameters"


def test_unet_gradient_flow():
    """Test that gradients flow through U-Net."""
    print("[TEST] U-Net Gradient Flow...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_unet_64x64(attn_resolutions={16, 8}, dropout=0.0).to(device)
    B, C, H, W = 2, 3, 64, 64
    
    x = torch.randn(B, C, H, W, device=device, requires_grad=False)
    t = torch.randint(0, 1000, (B,), device=device)
    
    eps_pred = model(x, t)
    loss = eps_pred.mean()
    loss.backward()
    
    # Check that gradients exist
    has_grad = False
    for param in model.parameters():
        if param.grad is not None:
            has_grad = True
            assert torch.isfinite(param.grad).all(), "Gradient contains NaN/Inf"
            break
    
    assert has_grad, "No gradients found in U-Net parameters"
    print(f"  [OK] Gradients flow correctly through U-Net")


def test_unet_different_resolutions():
    """Test U-Net with different input resolutions."""
    print("[TEST] U-Net Different Resolutions...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_unet_64x64(attn_resolutions={16, 8}).to(device)
    B, C = 2, 3
    
    resolutions = [(32, 32), (64, 64), (128, 128)]
    
    for H, W in resolutions:
        x = torch.randn(B, C, H, W, device=device)
        t = torch.randint(0, 1000, (B,), device=device)
        
        with torch.no_grad():
            eps_pred = model(x, t)
        
        assert eps_pred.shape == x.shape, f"Resolution {H}x{W} failed: {eps_pred.shape}"
        assert torch.isfinite(eps_pred).all(), f"Resolution {H}x{W} contains NaN/Inf"
        print(f"  [OK] Resolution {H}x{W}: {x.shape} -> {eps_pred.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("U-Net Backbone Sanity Checks")
    print("=" * 60)
    
    test_res_block()
    test_upsample_downsample()
    test_time_embeddings()
    test_unet_denoiser()
    test_unet_gradient_flow()
    test_unet_different_resolutions()
    
    print("\n[ALL GOOD] All U-Net backbone sanity checks passed!")
    print("=" * 60)

