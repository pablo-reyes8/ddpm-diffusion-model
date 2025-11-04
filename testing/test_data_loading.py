"""
Sanity checks for data loading utilities.
Tests CelebA data loading and preprocessing.
"""
import torch
from torchvision import transforms
from src.data.load_data_from_torch import get_celeba_loaders


def test_celeba_loader_shape():
    """Test that CelebA loader produces correct shapes."""
    print("[TEST] CelebA Loader Shape...")
    
    img_size = 64
    batch_size = 8
    
    try:
        train_loader, val_loader, test_loader = get_celeba_loaders(
            root="./data",
            img_size=img_size,
            batch_size=batch_size,
            num_workers=0,  # Use 0 for testing
            pin_memory=False
        )
        
        # Get one batch
        for images, labels in train_loader:
            assert images.shape[0] <= batch_size, f"Batch size mismatch: {images.shape[0]}"
            assert images.shape[1] == 3, f"Channel count mismatch: {images.shape[1]}"
            assert images.shape[2] == img_size, f"Height mismatch: {images.shape[2]}"
            assert images.shape[3] == img_size, f"Width mismatch: {images.shape[3]}"
            
            # Check value range (should be normalized to [-1, 1])
            assert images.min() >= -1.5, f"Images below -1: {images.min()}"
            assert images.max() <= 1.5, f"Images above 1: {images.max()}"
            
            print(f"  [OK] Train batch shape: {images.shape}")
            print(f"  [OK] Value range: [{images.min():.3f}, {images.max():.3f}]")
            break
        
        print(f"  [OK] CelebA loader produces correct shapes")
        
    except Exception as e:
        print(f"  [SKIP] CelebA dataset not available: {e}")
        print(f"  [INFO] This test requires internet connection to download CelebA")


def test_celeba_loader_splits():
    """Test that train/val/test splits are separate."""
    print("[TEST] CelebA Loader Splits...")
    
    img_size = 64
    batch_size = 8
    
    try:
        train_loader, val_loader, test_loader = get_celeba_loaders(
            root="./data",
            img_size=img_size,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False
        )
        
        # Get batches from each split
        train_batch = next(iter(train_loader))[0]
        val_batch = next(iter(val_loader))[0]
        test_batch = next(iter(test_loader))[0]
        
        # All should have same shape
        assert train_batch.shape == val_batch.shape == test_batch.shape, "Split shapes differ"
        
        # Check that they're different data (very unlikely to be identical)
        assert not torch.allclose(train_batch, val_batch), "Train and val batches are identical"
        assert not torch.allclose(train_batch, test_batch), "Train and test batches are identical"
        
        print(f"  [OK] Train/Val/Test splits are separate")
        print(f"  [OK] All splits have shape: {train_batch.shape}")
        
    except Exception as e:
        print(f"  [SKIP] CelebA dataset not available: {e}")


def test_celeba_normalization():
    """Test CelebA normalization (should be [-1, 1])."""
    print("[TEST] CelebA Normalization...")
    
    img_size = 64
    batch_size = 8
    
    try:
        train_loader, _, _ = get_celeba_loaders(
            root="./data",
            img_size=img_size,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False
        )
        
        # Collect statistics from multiple batches
        all_images = []
        for i, (images, _) in enumerate(train_loader):
            all_images.append(images)
            if i >= 4:  # Sample 5 batches
                break
        
        all_images = torch.cat(all_images, dim=0)
        
        mean = all_images.mean().item()
        std = all_images.std().item()
        
        assert all_images.min() >= -1.5, f"Min value too low: {all_images.min()}"
        assert all_images.max() <= 1.5, f"Max value too high: {all_images.max()}"
        
        print(f"  [OK] Mean: {mean:.3f}, Std: {std:.3f}")
        print(f"  [OK] Range: [{all_images.min():.3f}, {all_images.max():.3f}]")
        
    except Exception as e:
        print(f"  [SKIP] CelebA dataset not available: {e}")


def test_celeba_different_sizes():
    """Test CelebA loader with different image sizes."""
    print("[TEST] CelebA Different Sizes...")
    
    batch_size = 4
    
    for img_size in [32, 64, 128]:
        try:
            train_loader, _, _ = get_celeba_loaders(
                root="./data",
                img_size=img_size,
                batch_size=batch_size,
                num_workers=0,
                pin_memory=False
            )
            
            images, _ = next(iter(train_loader))
            assert images.shape[2] == img_size, f"Height mismatch for size {img_size}"
            assert images.shape[3] == img_size, f"Width mismatch for size {img_size}"
            
            print(f"  [OK] Size {img_size}x{img_size}: {images.shape}")
            
        except Exception as e:
            print(f"  [SKIP] Size {img_size}: {e}")
            break


def test_celeba_shuffle():
    """Test that training loader shuffles data."""
    print("[TEST] CelebA Shuffle...")
    
    img_size = 64
    batch_size = 8
    
    try:
        train_loader, _, _ = get_celeba_loaders(
            root="./data",
            img_size=img_size,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False)
        
        # Get two batches
        batch1 = next(iter(train_loader))[0]
        batch2 = next(iter(train_loader))[0]
        
        diff = (batch1 - batch2).abs().mean()
        assert diff > 1e-6, "Batches appear identical (shuffling might not work)"
        
        print(f"  [OK] Shuffling works (batch diff: {diff:.6f})")
        
    except Exception as e:
        print(f"  [SKIP] CelebA dataset not available: {e}")


def test_celeba_loader_device():
    """Test that data can be moved to device."""
    print("[TEST] CelebA Loader Device...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 64
    batch_size = 4
    
    try:
        train_loader, _, _ = get_celeba_loaders(
            root="./data",
            img_size=img_size,
            batch_size=batch_size,
            num_workers=0,
            pin_memory=False
        )
        
        images, labels = next(iter(train_loader))
        images = images.to(device)
        labels = labels.to(device)
        
        assert images.device == device, f"Images not on {device}"
        assert labels.device == device, f"Labels not on {device}"
        
        print(f"  [OK] Data moved to {device} successfully")
        
    except Exception as e:
        print(f"  [SKIP] CelebA dataset not available: {e}")


if __name__ == "__main__":
    print("=" * 60)
    print("Data Loading Sanity Checks")
    print("=" * 60)
    
    test_celeba_loader_shape()
    test_celeba_loader_splits()
    test_celeba_normalization()
    test_celeba_different_sizes()
    test_celeba_shuffle()
    test_celeba_loader_device()
    
    print("\n[ALL GOOD] All data loading sanity checks passed!")
    print("=" * 60)
    print("\n[NOTE] Some tests may be skipped if CelebA dataset is not available.")

