"""
Sanity checks for attention mechanisms.
Tests SinusoidalPosEmb, TimeMLP, GroupNorm, and AttnBlock.
"""
import torch
import torch.nn as nn
from src.model.attention import (
    SinusoidalPosEmb,
    TimeMLP,
    group_norm,
    AttnBlock
)


def test_sinusoidal_pos_emb():
    """Test sinusoidal positional embeddings."""
    print("[TEST] SinusoidalPosEmb...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different dimensions
    for dim in [128, 256, 512, 513]:  # Test odd dimension too
        pos_emb = SinusoidalPosEmb(dim).to(device)
        B = 8
        t = torch.randint(0, 1000, (B,), device=device)
        
        with torch.no_grad():
            emb = pos_emb(t)
        
        assert emb.shape == (B, dim), f"Embedding shape mismatch for dim={dim}: {emb.shape}"
        assert torch.isfinite(emb).all(), f"Embedding contains NaN/Inf for dim={dim}"
        
        # Check that different timesteps produce different embeddings
        t2 = torch.randint(0, 1000, (B,), device=device)
        emb2 = pos_emb(t2)
        assert not torch.allclose(emb, emb2), "Different timesteps should produce different embeddings"
        
        print(f"  [OK] dim={dim}: {t.shape} -> {emb.shape}")


def test_time_mlp():
    """Test TimeMLP projection."""
    print("[TEST] TimeMLP...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    in_dim, out_dim = 512, 512
    mlp = TimeMLP(in_dim, out_dim).to(device)
    
    B = 8
    t_emb = torch.randn(B, in_dim, device=device)
    
    with torch.no_grad():
        out = mlp(t_emb)
    
    assert out.shape == (B, out_dim), f"TimeMLP output shape: {out.shape}"
    assert torch.isfinite(out).all(), "TimeMLP output contains NaN/Inf"
    
    # Test gradient flow
    t_emb.requires_grad_(True)
    out = mlp(t_emb)
    loss = out.mean()
    loss.backward()
    
    has_grad = any(p.grad is not None for p in mlp.parameters())
    assert has_grad, "No gradients in TimeMLP"
    
    print(f"  [OK] TimeMLP: {t_emb.shape} -> {out.shape}")


def test_group_norm():
    """Test GroupNorm creation and properties."""
    print("[TEST] GroupNorm...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test different channel counts
    for channels in [32, 64, 128, 256]:
        gn = group_norm(channels).to(device)
        B, H, W = 4, 32, 32
        x = torch.randn(B, channels, H, W, device=device)
        
        with torch.no_grad():
            out = gn(x)
        
        assert out.shape == x.shape, f"GroupNorm shape mismatch: {out.shape}"
        assert torch.isfinite(out).all(), "GroupNorm output contains NaN/Inf"
        
        print(f"  [OK] channels={channels}: {x.shape} -> {out.shape}")


def test_attn_block():
    """Test attention block forward pass."""
    print("[TEST] AttnBlock...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test at different resolutions
    resolutions = [
        (8, 8, 64, 4),   # H, W, channels, num_heads
        (16, 16, 128, 4),
        (32, 32, 256, 8),
    ]
    
    for H, W, channels, num_heads in resolutions:
        attn = AttnBlock(channels, num_heads=num_heads, head_dim=64).to(device)
        B = 4
        x = torch.randn(B, channels, H, W, device=device)
        
        with torch.no_grad():
            out = attn(x)
        
        assert out.shape == x.shape, f"AttnBlock shape mismatch at {H}x{W}: {out.shape}"
        assert torch.isfinite(out).all(), f"AttnBlock output contains NaN/Inf at {H}x{W}"
        
        # Check residual connection (output should be close to input + small change)
        diff = (out - x).abs().mean()
        assert diff > 0, "AttnBlock should modify input"
        print(f"  [OK] {H}x{W}, channels={channels}, heads={num_heads}: {x.shape} -> {out.shape}")


def test_attn_block_gradient():
    """Test gradient flow through attention block."""
    print("[TEST] AttnBlock Gradient Flow...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    channels, num_heads = 128, 4
    attn = AttnBlock(channels, num_heads=num_heads, head_dim=64).to(device)
    
    B, H, W = 4, 16, 16
    x = torch.randn(B, channels, H, W, device=device, requires_grad=False)
    
    out = attn(x)
    loss = out.mean()
    loss.backward()
    
    # Check gradients
    has_grad = False
    for param in attn.parameters():
        if param.grad is not None:
            has_grad = True
            assert torch.isfinite(param.grad).all(), "Gradient contains NaN/Inf"
            break
    
    assert has_grad, "No gradients in AttnBlock"
    print(f"  [OK] Gradients flow correctly through AttnBlock")


def test_attn_block_consistency():
    """Test that attention block is deterministic with same seed."""
    print("[TEST] AttnBlock Consistency...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    channels, num_heads = 128, 4
    attn = AttnBlock(channels, num_heads=num_heads, head_dim=64).to(device)
    
    B, H, W = 4, 16, 16
    
    # Set seed and create input
    torch.manual_seed(42)
    x1 = torch.randn(B, channels, H, W, device=device)
    
    # First forward pass
    with torch.no_grad():
        out1 = attn(x1)
    
    # Second forward pass with same input
    torch.manual_seed(42)
    x2 = torch.randn(B, channels, H, W, device=device)
    
    with torch.no_grad():
        out2 = attn(x2)
    
    # Should be identical (deterministic)
    assert torch.allclose(out1, out2, atol=1e-6), "AttnBlock should be deterministic"
    assert torch.allclose(x1, x2, atol=1e-6), "Inputs should be identical"
    
    print(f"  [OK] AttnBlock is deterministic")


def test_attn_block_different_heads():
    """Test attention block with different number of heads."""
    print("[TEST] AttnBlock Different Heads...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    channels = 128
    B, H, W = 4, 16, 16
    x = torch.randn(B, channels, H, W, device=device)
    
    for num_heads in [1, 2, 4, 8]:
        head_dim = 64
        attn = AttnBlock(channels, num_heads=num_heads, head_dim=head_dim).to(device)
        
        with torch.no_grad():
            out = attn(x)
        
        assert out.shape == x.shape, f"Shape mismatch with {num_heads} heads"
        assert torch.isfinite(out).all(), f"NaN/Inf with {num_heads} heads"
        
        print(f"  [OK] {num_heads} heads: {x.shape} -> {out.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("Attention Components Sanity Checks")
    print("=" * 60)
    
    test_sinusoidal_pos_emb()
    test_time_mlp()
    test_group_norm()
    test_attn_block()
    test_attn_block_gradient()
    test_attn_block_consistency()
    test_attn_block_different_heads()
    
    print("\n[ALL GOOD] All attention sanity checks passed!")
    print("=" * 60)

