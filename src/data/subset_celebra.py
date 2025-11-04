import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

def create_subset_dataloader(
    dataset,
    max_samples=100_000,
    batch_size=64,
    num_workers=4,
    img_size=None,
    seed_subset=2025,
    seed_shuffle=1337
):
    """
    Crea un DataLoader con un subconjunto aleatorio del dataset original.

    Parámetros
    ----------
    dataset : torch.utils.data.Dataset
        Dataset original (o dataset base del DataLoader).
    max_samples : int, opcional
        Número máximo de ejemplos a incluir (por defecto 100_000).
    batch_size : int, opcional
        Tamaño del batch para el nuevo DataLoader.
    num_workers : int, opcional
        Número de procesos de carga paralelos.
    img_size : int, opcional
        Tamaño de imagen (solo usado para imprimir información).
    seed_subset : int, opcional
        Semilla para la selección aleatoria del subconjunto.
    seed_shuffle : int, opcional
        Semilla para el barajado del DataLoader.

    Retorna
    -------
    subset : torch.utils.data.Subset
        Subconjunto del dataset original.
    loader : torch.utils.data.DataLoader
        DataLoader listo para entrenamiento con el subconjunto.
    """
    N = min(max_samples, len(dataset))
    rng = np.random.default_rng(seed_subset)
    subset_indices = rng.choice(len(dataset), size=N, replace=False)
    subset = Subset(dataset, subset_indices.tolist())

    g = torch.Generator()
    g.manual_seed(seed_shuffle)

    loader = DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g
    )

    print(f"[OK] Subset listo: {len(subset)} muestras (de {len(dataset)})")
    print(f"[OK] DataLoader listo: batch_size={batch_size}, img_size={img_size}")

    return subset, loader
