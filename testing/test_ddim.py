"""
Sanity checks for DDIM (Denoising Diffusion Implicit Models) sampling.
Tests DDIM sampling steps and full inference pipeline.
"""
import torch
import torch.nn as nn
from src.model.difussion_class import Diffusion
from src.model.unet_backbone import build_unet_64x64


def test_ddim_single_step():
    """Test a single DDIM sampling step."""
    print("[TEST] DDIM Single Step...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    diffusion = Diffusion(T=1000, schedule="linear").to(device)
    model = build_unet_64x64(attn_resolutions={16, 8}).to(device)
    
    B, C, H, W = 4, 3, 64, 64
    
    # Create noisy image at timestep t
    t = torch.full((B,), 500, device=device, dtype=torch.long)
    t_prev = torch.full((B,), 450, device=device, dtype=torch.long)
    
    x_t = torch.randn(B, C, H, W, device=device)
    
    def model_fn(x, t):
        return model(x, t)
    
    with torch.no_grad():
        x_prev = diffusion.p_sample_step_ddim(
            model_fn, x_t, t, t_prev, eta=0.0
        )
    
    assert x_prev.shape == x_t.shape, f"DDIM step shape mismatch: {x_prev.shape}"
    assert torch.isfinite(x_prev).all(), "DDIM step output contains NaN/Inf"
    
    print(f"  [OK] DDIM step: t={t[0].item()} -> t_prev={t_prev[0].item()}, {x_t.shape} -> {x_prev.shape}")


def test_ddim_deterministic():
    """Test that DDIM with eta=0.0 is deterministic."""
    print("[TEST] DDIM Deterministic (eta=0.0)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    diffusion = Diffusion(T=1000, schedule="linear").to(device)
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    model.eval()
    
    B, C, H, W = 2, 3, 64, 64
    
    # Set seed
    torch.manual_seed(123)
    x_t1 = torch.randn(B, C, H, W, device=device)
    t = torch.full((B,), 500, device=device, dtype=torch.long)
    t_prev = torch.full((B,), 450, device=device, dtype=torch.long)
    
    def model_fn(x, t):
        return model(x, t)
    
    with torch.no_grad():
        x_prev1 = diffusion.p_sample_step_ddim(model_fn, x_t1, t, t_prev, eta=0.0)
    
    # Repeat with same seed
    torch.manual_seed(123)
    x_t2 = torch.randn(B, C, H, W, device=device)
    
    with torch.no_grad():
        x_prev2 = diffusion.p_sample_step_ddim(model_fn, x_t2, t, t_prev, eta=0.0)
    
    assert torch.allclose(x_t1, x_t2, atol=1e-6), "Inputs should be identical"
    assert torch.allclose(x_prev1, x_prev2, atol=1e-5), "DDIM with eta=0.0 should be deterministic"
    
    print(f"  [OK] DDIM deterministic sampling verified")


def test_ddim_stochastic():
    """Test that DDIM with eta>0 adds stochasticity."""
    print("[TEST] DDIM Stochastic (eta>0)...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    diffusion = Diffusion(T=1000, schedule="linear").to(device)
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    model.eval()
    
    B, C, H, W = 2, 3, 64, 64
    
    x_t = torch.randn(B, C, H, W, device=device)
    t = torch.full((B,), 500, device=device, dtype=torch.long)
    t_prev = torch.full((B,), 450, device=device, dtype=torch.long)
    
    def model_fn(x, t):
        return model(x, t)
    
    # Run twice with same input but different randomness
    with torch.no_grad():
        x_prev1 = diffusion.p_sample_step_ddim(model_fn, x_t, t, t_prev, eta=1.0)
        x_prev2 = diffusion.p_sample_step_ddim(model_fn, x_t, t, t_prev, eta=1.0)
    
    # Should be different due to random noise
    diff = (x_prev1 - x_prev2).abs().mean()
    assert diff > 1e-4, f"DDIM with eta>0 should be stochastic, diff={diff:.6f}"
    
    print(f"  [OK] DDIM stochastic sampling verified (diff={diff:.6f})")


def test_ddim_vs_ddpm_step():
    """Compare DDIM and DDPM single steps."""
    print("[TEST] DDIM vs DDPM Step Comparison...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    diffusion = Diffusion(T=1000, schedule="linear").to(device)
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    model.eval()
    
    B, C, H, W = 2, 3, 64, 64
    
    torch.manual_seed(42)
    x_t = torch.randn(B, C, H, W, device=device)
    t = torch.full((B,), 500, device=device, dtype=torch.long)
    
    def model_fn(x, t):
        return model(x, t)
    
    # DDPM step
    torch.manual_seed(42)
    with torch.no_grad():
        x_prev_ddpm = diffusion.p_sample_step(model_fn, x_t, t, eta=1.0)
    
    # DDIM step (t_prev = t - 1)
    torch.manual_seed(42)
    t_prev = torch.full((B,), 499, device=device, dtype=torch.long)
    with torch.no_grad():
        x_prev_ddim = diffusion.p_sample_step_ddim(model_fn, x_t, t, t_prev, eta=0.0)
    
    # Both should produce valid outputs
    assert torch.isfinite(x_prev_ddpm).all(), "DDPM step contains NaN/Inf"
    assert torch.isfinite(x_prev_ddim).all(), "DDIM step contains NaN/Inf"
    assert x_prev_ddpm.shape == x_prev_ddim.shape, "Shape mismatch between DDPM and DDIM"
    
    print(f"  [OK] DDPM: {x_t.shape} -> {x_prev_ddpm.shape}")
    print(f"  [OK] DDIM: {x_t.shape} -> {x_prev_ddim.shape}")


def test_ddim_multiple_steps():
    """Test multiple DDIM steps in sequence."""
    print("[TEST] DDIM Multiple Steps...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    diffusion = Diffusion(T=1000, schedule="linear").to(device)
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    model.eval()
    
    B, C, H, W = 2, 3, 64, 64
    
    # Start from noise
    torch.manual_seed(42)
    x = torch.randn(B, C, H, W, device=device)
    
    def model_fn(x, t):
        return model(x, t)
    
    # Run 10 steps
    steps = 10
    timesteps = torch.linspace(999, 0, steps + 1, device=device).long()
    
    with torch.no_grad():
        for i in range(steps):
            t = timesteps[i]
            t_prev = timesteps[i + 1]
            x = diffusion.p_sample_step_ddim(model_fn, x, t, t_prev, eta=0.0)
            
            assert torch.isfinite(x).all(), f"NaN/Inf at step {i}"
            assert x.shape == (B, C, H, W), f"Shape changed at step {i}: {x.shape}"
    
    print(f"  [OK] Completed {steps} DDIM steps successfully")


def test_ddim_different_schedules():
    """Test DDIM with different diffusion schedules."""
    print("[TEST] DDIM Different Schedules...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    schedules = ["linear", "cosine"]
    
    for schedule in schedules:
        diffusion = Diffusion(T=1000, schedule=schedule).to(device)
        model = build_unet_64x64(attn_resolutions=set()).to(device)
        model.eval()
        
        B, C, H, W = 2, 3, 64, 64
        x_t = torch.randn(B, C, H, W, device=device)
        t = torch.full((B,), 500, device=device, dtype=torch.long)
        t_prev = torch.full((B,), 450, device=device, dtype=torch.long)
        
        def model_fn(x, t):
            return model(x, t)
        
        with torch.no_grad():
            x_prev = diffusion.p_sample_step_ddim(model_fn, x_t, t, t_prev, eta=0.0)
        
        assert torch.isfinite(x_prev).all(), f"NaN/Inf with {schedule} schedule"
        assert x_prev.shape == x_t.shape, f"Shape mismatch with {schedule} schedule"
        
        print(f"  [OK] Schedule '{schedule}': {x_t.shape} -> {x_prev.shape}")


def test_ddim_edge_cases():
    """Test DDIM edge cases (t=0, t=T-1, etc.)."""
    print("[TEST] DDIM Edge Cases...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    diffusion = Diffusion(T=1000, schedule="linear").to(device)
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    model.eval()
    
    B, C, H, W = 2, 3, 64, 64
    
    def model_fn(x, t):
        return model(x, t)
    
    # Test t=0 (final step)
    x_t = torch.randn(B, C, H, W, device=device)
    t = torch.zeros((B,), device=device, dtype=torch.long)
    t_prev = torch.zeros((B,), device=device, dtype=torch.long)
    
    with torch.no_grad():
        x_prev = diffusion.p_sample_step_ddim(model_fn, x_t, t, t_prev, eta=0.0)
    
    assert torch.isfinite(x_prev).all(), "Edge case t=0 failed"
    print(f"  [OK] Edge case t=0: {x_t.shape} -> {x_prev.shape}")
    
    # Test t=T-1 (first step)
    t = torch.full((B,), 999, device=device, dtype=torch.long)
    t_prev = torch.full((B,), 998, device=device, dtype=torch.long)
    
    with torch.no_grad():
        x_prev = diffusion.p_sample_step_ddim(model_fn, x_t, t, t_prev, eta=0.0)
    
    assert torch.isfinite(x_prev).all(), "Edge case t=T-1 failed"
    print(f"  [OK] Edge case t=T-1: {x_t.shape} -> {x_prev.shape}")


if __name__ == "__main__":
    print("=" * 60)
    print("DDIM Sanity Checks")
    print("=" * 60)
    
    test_ddim_single_step()
    test_ddim_deterministic()
    test_ddim_stochastic()
    test_ddim_vs_ddpm_step()
    test_ddim_multiple_steps()
    test_ddim_different_schedules()
    test_ddim_edge_cases()
    
    print("\n[ALL GOOD] All DDIM sanity checks passed!")
    print("=" * 60)

