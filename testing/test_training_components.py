"""
Sanity checks for training components.
Tests EMA, checkpoints, and training utilities.
"""
import torch
import torch.nn as nn
import os
import tempfile
from src.training_loops.ema import EMA
from src.training_loops.chekpoints import save_ckpt, load_ckpt
from src.model.unet_backbone import build_unet_64x64


def test_ema_initialization():
    """Test EMA initialization."""
    print("[TEST] EMA Initialization...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    decay = 0.999
    
    ema = EMA(model, decay=decay, device=device)
    
    assert ema.decay == decay, f"EMA decay mismatch: {ema.decay} != {decay}"
    assert len(ema.shadow) > 0, "EMA shadow should have parameters"
    
    # Check that shadow params match model params
    for i, param in enumerate(model.parameters()):
        if param.requires_grad:
            assert ema.shadow[i] is not None, f"Shadow param {i} is None"
            assert ema.shadow[i].shape == param.shape, f"Shadow shape mismatch at {i}"
            assert torch.allclose(ema.shadow[i], param), f"Shadow not initialized correctly at {i}"
    
    print(f"  [OK] EMA initialized with {len([s for s in ema.shadow if s is not None])} parameters")


def test_ema_update():
    """Test EMA update mechanism."""
    print("[TEST] EMA Update...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    decay = 0.9  # Use lower decay for faster testing
    
    ema = EMA(model, decay=decay, device=device)
    
    # Get initial shadow values
    initial_shadow = [s.clone() if s is not None else None for s in ema.shadow]
    
    for param in model.parameters():
        if param.requires_grad:
            param.data += 1.0
    
    ema.update(model)
    
    # Check that shadow moved towards new values
    shadow_idx = 0
    for param in model.parameters():
        if param.requires_grad:
            shadow = ema.shadow[shadow_idx]
            initial = initial_shadow[shadow_idx]
            
            # Shadow should be between initial and new value
            diff_initial = (shadow - initial).abs().mean()
            diff_new = (shadow - param).abs().mean()
            
            assert diff_initial > 0, "Shadow should have changed"
            assert diff_new < diff_initial, "Shadow should be closer to new value"
            
            shadow_idx += 1
    
    print(f"  [OK] EMA update mechanism works correctly")


def test_ema_copy_to():
    """Test copying EMA values back to model."""
    print("[TEST] EMA Copy To Model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    decay = 0.999
    
    ema = EMA(model, decay=decay, device=device)
    
    # Modify model parameters
    original_params = {}
    for name, param in model.named_parameters():
        if param.requires_grad:
            original_params[name] = param.data.clone()
            param.data += 10.0
    
    # Update EMA multiple times
    for _ in range(10):
        ema.update(model)
    
    # Copy EMA back to model
    backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ema.copy_to(model)
    
    # Check that model params match EMA shadow
    shadow_idx = 0
    for param in model.parameters():
        if param.requires_grad:
            shadow = ema.shadow[shadow_idx]
            assert torch.allclose(param.data, shadow, atol=1e-5), "Model params don't match EMA shadow"
            shadow_idx += 1
    
    # Restore original
    model.load_state_dict(backup)
    
    print(f"  [OK] EMA copy_to works correctly")


def test_ema_state_dict():
    """Test EMA state dict save/load."""
    print("[TEST] EMA State Dict...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    decay = 0.999
    
    ema1 = EMA(model, decay=decay, device=device)
    
    # Update EMA
    for _ in range(5):
        ema1.update(model)
    
    state = ema1.state_dict()
    
    # Create new EMA and load state
    ema2 = EMA(model, decay=0.5, device=device)
    ema2.load_state_dict(state)
    
    assert ema2.decay == decay, "Decay not loaded correctly"
    assert len(ema2.shadow) == len(ema1.shadow), "Shadow length mismatch"
    
    # Check shadow values match
    for s1, s2 in zip(ema1.shadow, ema2.shadow):
        if s1 is not None and s2 is not None:
            assert torch.allclose(s1, s2), "Shadow values don't match"
    
    print(f"  [OK] EMA state dict save/load works correctly")


def test_checkpoint_save_load():
    """Test checkpoint save and load."""
    print("[TEST] Checkpoint Save/Load...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ema = EMA(model, decay=0.999, device=device)
    
    # Create a dummy scaler object that has state_dict method
    class DummyScaler:
        def state_dict(self):
            return {}
        def load_state_dict(self, state):
            pass
    
    scaler = DummyScaler()
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = os.path.join(tmpdir, "test_checkpoint.pt")
        
        # Save checkpoint
        save_ckpt(
            ckpt_path,
            model,
            optimizer,
            scaler=scaler,
            ema=ema,
            step=100,
            extra={"epoch": 10, "test_key": "test_value"})
        
        assert os.path.exists(ckpt_path), "Checkpoint file not created"
        
        # Create new model and load
        model2 = build_unet_64x64(attn_resolutions=set()).to(device)
        optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-4)
        ema2 = EMA(model2, decay=0.999, device=device)
        scaler2 = DummyScaler()
        
        step_loaded, extra = load_ckpt(
            ckpt_path,
            model2,
            optimizer=optimizer2,
            scaler=scaler2,
            ema=ema2,
            map_location=device)
        
        assert step_loaded == 100, f"Step not loaded correctly: {step_loaded}"
        assert extra["epoch"] == 10, "Extra epoch not loaded"
        assert extra["test_key"] == "test_value", "Extra values not loaded"
        
        # Check model parameters match
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.allclose(p1, p2), f"Model param {n1} doesn't match after load"
        
        print(f"  [OK] Checkpoint save/load works correctly")


def test_ema_with_training_step():
    """Test EMA integration with training step."""
    print("[TEST] EMA with Training Step...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    ema = EMA(model, decay=0.9, device=device)
    
    B, C, H, W = 2, 3, 64, 64
    x = torch.randn(B, C, H, W, device=device)
    t = torch.randint(0, 1000, (B,), device=device)
    
    # Training step
    optimizer.zero_grad()
    eps_pred = model(x, t)
    loss = eps_pred.mean()
    loss.backward()
    optimizer.step()
    
    ema.update(model)
    
    # Verify EMA shadow is different from model (after update)
    shadow_idx = 0
    for param in model.parameters():
        if param.requires_grad:
            shadow = ema.shadow[shadow_idx]
            # After update, shadow should be close but not identical
            diff = (shadow - param).abs().mean()
            assert diff > 0, "EMA shadow should differ from model"
            shadow_idx += 1
    
    print(f"  [OK] EMA integrates correctly with training")


def test_ema_multiple_updates():
    """Test EMA with multiple updates."""
    print("[TEST] EMA Multiple Updates...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = build_unet_64x64(attn_resolutions=set()).to(device)
    decay = 0.9
    ema = EMA(model, decay=decay, device=device)
    
    initial_shadow = ema.shadow[0].clone() if ema.shadow[0] is not None else None
    
    for step in range(10):
        for param in model.parameters():
            if param.requires_grad:
                param.data += 0.1
        ema.update(model)
    
    # Check that shadow has moved towards new values
    final_shadow = ema.shadow[0]
    if initial_shadow is not None and final_shadow is not None:
        diff = (final_shadow - initial_shadow).abs().mean()
        assert diff > 0.1, f"EMA shadow should have moved significantly: {diff}"
    
    print(f"  [OK] EMA with multiple updates works correctly")


if __name__ == "__main__":
    print("=" * 60)
    print("Training Components Sanity Checks")
    print("=" * 60)
    
    test_ema_initialization()
    test_ema_update()
    test_ema_copy_to()
    test_ema_state_dict()
    test_checkpoint_save_load()
    test_ema_with_training_step()
    test_ema_multiple_updates()
    
    print("\n[ALL GOOD] All training components sanity checks passed!")
    print("=" * 60)

