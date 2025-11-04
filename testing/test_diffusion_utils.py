"""
Sanity checks for diffusion utility functions.
Tests beta schedules, extract function, and schedule properties.
"""
import torch
import math
from src.model.difussion_utils import (
    extract,
    beta_schedule_linear,
    beta_schedule_cosine,
    _alpha_bar_cosine
)


def test_extract_function():
    """Test the extract function for indexing tensors."""
    print("[TEST] extract() function...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    T = 1000
    a = torch.arange(T, dtype=torch.float32, device=device)
    
    # Test with different shapes
    shapes = [
        (4, 3, 64, 64),  # Images
        (8, 128),        # Features
        (2,),            # 1D
    ]
    
    for x_shape in shapes:
        B = x_shape[0]
        t = torch.randint(0, T, (B,), device=device)
        
        extracted = extract(a, t, torch.Size(x_shape))
        
        # Check shape
        expected_shape = (B,) + (1,) * (len(x_shape) - 1)
        assert extracted.shape == expected_shape, f"Shape mismatch: {extracted.shape} != {expected_shape}"
        
        # Check values
        for i in range(B):
            expected_val = a[t[i]].item()
            actual_val = extracted[i].item()
            assert abs(expected_val - actual_val) < 1e-6, f"Value mismatch at index {i}"
        
        print(f"  [OK] Shape {x_shape}: extracted shape = {extracted.shape}")


def test_extract_edge_cases():
    """Test extract function with edge cases."""
    print("[TEST] extract() Edge Cases...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    T = 1000
    a = torch.arange(T, dtype=torch.float32, device=device)
    
    # Test t=0
    t_zero = torch.zeros((4,), device=device, dtype=torch.long)
    x_shape = torch.Size((4, 3, 64, 64))
    extracted = extract(a, t_zero, x_shape)
    assert extracted.shape == (4, 1, 1, 1), "Edge case t=0 failed"
    assert torch.allclose(extracted, torch.zeros_like(extracted)), "t=0 should extract a[0]"
    
    # Test t=T-1
    t_max = torch.full((4,), T-1, device=device, dtype=torch.long)
    extracted = extract(a, t_max, x_shape)
    assert torch.allclose(extracted, torch.full_like(extracted, T-1)), "t=T-1 should extract a[T-1]"
    
    # Test clamping (t < 0)
    t_neg = torch.full((4,), -1, device=device, dtype=torch.long)
    extracted = extract(a, t_neg, x_shape)
    assert torch.allclose(extracted, torch.zeros_like(extracted)), "t<0 should clamp to 0"
    
    print(f"  [OK] Edge cases handled correctly")


def test_beta_schedule_linear():
    """Test linear beta schedule."""
    print("[TEST] beta_schedule_linear()...")
    
    T = 1000
    beta_min, beta_max = 1e-4, 2e-2
    
    betas = beta_schedule_linear(T, beta_min, beta_max)
    
    assert len(betas) == T, f"Beta schedule length mismatch: {len(betas)} != {T}"
    assert betas[0] == beta_min, f"First beta should be {beta_min}, got {betas[0]}"
    assert abs(betas[-1] - beta_max) < 1e-6, f"Last beta should be {beta_max}, got {betas[-1]}"
    
    # Check monotonicity
    for i in range(1, T):
        assert betas[i] > betas[i-1], f"Betas should be increasing: {betas[i]} <= {betas[i-1]}"
    
    # Check range
    assert (betas >= beta_min).all(), "Betas below minimum"
    assert (betas <= beta_max).all(), "Betas above maximum"
    
    print(f"  [OK] Linear schedule: {len(betas)} steps, range [{betas[0]:.6f}, {betas[-1]:.6f}]")


def test_beta_schedule_cosine():
    """Test cosine beta schedule."""
    print("[TEST] beta_schedule_cosine()...")
    
    T = 1000
    s = 0.008
    
    betas = beta_schedule_cosine(T, s=s)
    
    assert len(betas) == T, f"Beta schedule length mismatch: {len(betas)} != {T}"
    
    # Check range (betas should be in [0, 1))
    assert (betas >= 0).all(), "Negative betas in cosine schedule"
    assert (betas < 1).all(), "Betas >= 1 in cosine schedule"
    
    # Cosine schedule should start small and end larger (but not monotonically increasing)
    # First beta should be very small
    assert betas[0] < 0.01, f"First beta should be small: {betas[0]}"
    
    print(f"  [OK] Cosine schedule: {len(betas)} steps, range [{betas[0]:.6f}, {betas[-1]:.6f}]")


def test_alpha_bar_cosine():
    """Test alpha_bar cosine function."""
    print("[TEST] _alpha_bar_cosine()...")
    
    s = 0.008
    
    # Test at t=0
    t0 = torch.tensor([0.0])
    alpha_bar_0 = _alpha_bar_cosine(t0, s=s)
    assert alpha_bar_0.item() > 0.9, f"alpha_bar(0) should be close to 1: {alpha_bar_0.item()}"
    
    # Test at t=1
    t1 = torch.tensor([1.0])
    alpha_bar_1 = _alpha_bar_cosine(t1, s=s)
    assert alpha_bar_1.item() > 0, f"alpha_bar(1) should be positive: {alpha_bar_1.item()}"
    assert alpha_bar_1.item() < alpha_bar_0.item(), "alpha_bar should decrease from 0 to 1"
    
    # Test monotonicity
    t = torch.linspace(0, 1, 100)
    alpha_bars = _alpha_bar_cosine(t, s=s)
    for i in range(1, len(alpha_bars)):
        assert alpha_bars[i] <= alpha_bars[i-1], "alpha_bar should be non-increasing"
    
    print(f"  [OK] alpha_bar_cosine: alpha_bar(0)={alpha_bar_0.item():.6f}, alpha_bar(1)={alpha_bar_1.item():.6f}")


def test_beta_to_alpha_conversion():
    """Test conversion from betas to alphas."""
    print("[TEST] Beta to Alpha Conversion...")
    
    T = 1000
    betas = beta_schedule_linear(T, 1e-4, 2e-2)
    
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    
    # Check properties
    assert len(alphas) == T, "Alphas length mismatch"
    assert (alphas > 0).all() and (alphas < 1).all(), "Alphas should be in (0, 1)"
    
    # Check alphas_cumprod
    assert len(alphas_cumprod) == T, "Alphas cumprod length mismatch"
    assert alphas_cumprod[0] == alphas[0], "First cumprod should equal first alpha"
    assert alphas_cumprod[-1] < alphas_cumprod[0], "Cumprod should decrease"
    assert (alphas_cumprod > 0).all() and (alphas_cumprod <= 1).all(), "Cumprod should be in (0, 1]"
    
    # Check monotonicity
    for i in range(1, T):
        assert alphas_cumprod[i] < alphas_cumprod[i-1], "Cumprod should be decreasing"
    
    print(f"  [OK] Conversion: alphas in [{alphas.min():.6f}, {alphas.max():.6f}]")
    print(f"  [OK] Cumprod: [{alphas_cumprod[0]:.6f}, {alphas_cumprod[-1]:.6f}]")


def test_schedule_comparison():
    """Compare linear and cosine schedules."""
    print("[TEST] Schedule Comparison...")
    
    T = 1000
    
    betas_linear = beta_schedule_linear(T, 1e-4, 2e-2)
    betas_cosine = beta_schedule_cosine(T, s=0.008)
    
    # Both should have same length
    assert len(betas_linear) == len(betas_cosine), "Schedule lengths should match"
    
    # Linear should be more uniform
    linear_range = betas_linear[-1] - betas_linear[0]
    cosine_range = betas_cosine.max() - betas_cosine.min()
    
    # Cosine typically has smaller range
    print(f"  [OK] Linear range: {linear_range:.6f}")
    print(f"  [OK] Cosine range: {cosine_range:.6f}")
    print(f"  [OK] Linear: [{betas_linear[0]:.6f}, {betas_linear[-1]:.6f}]")
    print(f"  [OK] Cosine: [{betas_cosine.min():.6f}, {betas_cosine.max():.6f}]")


def test_extract_broadcasting():
    """Test extract function broadcasting behavior."""
    print("[TEST] extract() Broadcasting...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    T = 1000
    a = torch.arange(T, dtype=torch.float32, device=device)
    
    B = 4
    t = torch.randint(0, T, (B,), device=device)
    
    # Test with 4D tensor (images)
    x_4d = torch.randn(B, 3, 64, 64, device=device)
    extracted_4d = extract(a, t, x_4d.shape)
    assert extracted_4d.shape == (B, 1, 1, 1), f"4D broadcasting failed: {extracted_4d.shape}"
    
    # Test with 2D tensor
    x_2d = torch.randn(B, 128, device=device)
    extracted_2d = extract(a, t, x_2d.shape)
    assert extracted_2d.shape == (B, 1), f"2D broadcasting failed: {extracted_2d.shape}"
    
    # Test multiplication (should broadcast correctly)
    result = x_4d * extracted_4d
    assert result.shape == x_4d.shape, "Broadcasting multiplication failed"
    
    print(f"  [OK] Broadcasting works correctly for different tensor shapes")


if __name__ == "__main__":
    print("=" * 60)
    print("Diffusion Utils Sanity Checks")
    print("=" * 60)
    
    test_extract_function()
    test_extract_edge_cases()
    test_beta_schedule_linear()
    test_beta_schedule_cosine()
    test_alpha_bar_cosine()
    test_beta_to_alpha_conversion()
    test_schedule_comparison()
    test_extract_broadcasting()
    
    print("\n[ALL GOOD] All diffusion utils sanity checks passed!")
    print("=" * 60)

