"""Random-number seeding shared by generation runners.

Importing this module does not load the generation graph or any model.
"""

from __future__ import annotations

import random


def seed_everything(seed: int) -> int:
    """Seed Python, NumPy, and PyTorch's CPU and CUDA random generators.

    Seeding makes repeated sampling reproducible with the same runtime and
    model settings; it does not enable deterministic CUDA algorithms.
    """
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed also seeds all CUDA devices, including lazy contexts.
    torch.manual_seed(seed)
    return seed
