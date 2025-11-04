import math
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from PIL import ImageFile
import datasets as hfds
from datasets import load_dataset

def build_hf_image_loader(
    dataset_name: str = "eurecom-ds/celeba-hq-256",
    split: str = "default",            # si falla, cae a "train"
    img_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 2,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: int = 2,
    seed: int = 1337,
    do_smoke_test: bool = True,
):
    """
    Crea un DataLoader robusto para datasets de HuggingFace con columna 'image' (PIL).
    Devuelve: (train_loader, ds)

    - Decodifica PIL con hfds.Image(decode=True)
    - Transforms en CPU (Resize → ToTensor → Normalize a [-1,1])
    - set_transform con tolerancia a errores por imagen
    - Collate controlado (stack + etiqueta dummy)
    - Workers con seeds disjuntas
    """

    ImageFile.LOAD_TRUNCATED_IMAGES = True

    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    tfm = T.Compose([
        T.Resize((img_size, img_size)), 
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),])

    try:
        ds = load_dataset(dataset_name, split=split)
    except Exception:
        ds = load_dataset(dataset_name, split="train")
    ds = ds.cast_column("image", hfds.Image(decode=True))  

    def safe_batched_transform(examples):
        imgs = examples["image"]
        out = []
        for img in imgs:
            try:
                out.append(tfm(img.convert("RGB")))
            except Exception as e:
                name = getattr(img, "filename", "unknown")
                print(f"[WARN] Imagen defectuosa: {name} | {e}")
                out.append(torch.zeros(3, img_size, img_size))
        return {"image": out}

    ds.set_transform(safe_batched_transform)

    def collate_fn(batch):
        x = torch.stack([b["image"] for b in batch], dim=0)
        y = torch.zeros(len(batch), dtype=torch.long)
        return x, y

    def w_init(worker_id):
        s = (torch.initial_seed() % 2**32)
        random.seed(s); np.random.seed(s % (2**32 - 1))

    train_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        worker_init_fn=w_init,
        collate_fn=collate_fn,
        timeout=0,
        drop_last=False,  
    )

    if do_smoke_test:
        print(f"[OK] Total imágenes: {len(ds)}")
        xb, yb = next(iter(train_loader))
        vmin, vmax = xb.min().item(), xb.max().item()
        print(f"[OK] Batch: {xb.shape} {xb.dtype} range=({vmin:.3f},{vmax:.3f})")

    return train_loader, ds
