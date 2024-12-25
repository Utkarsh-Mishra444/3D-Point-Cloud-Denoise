### Transformer with decoder


import torch
import torch.nn as nn
from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import models.train_utils as train_utils
from models.modules import Attention

from ..pvcnn import (
    Pnet2Stage,
    PVCData,
    SharedMLP,
    Swish,
    create_fp_components,
    create_mlp_components,
)

class HashEncoding(nn.Module):
    def __init__(
        self,
        dim: int,
        num_levels: int = 16,
        per_level_scale: float = 2.0,
        base_resolution: int = 16,
        log2_hashmap_size: int = 22,
    ):  
        print("using Hash encoding with interpolation run")
        super().__init__()
        self.dim = dim  # Desired output dimension
        print(f"Output dimension (dim): {self.dim}")

        self.num_levels = num_levels
        print(f"Number of levels (num_levels): {self.num_levels}")

        self.per_level_scale = per_level_scale
        print(f"Per level scale (per_level_scale): {self.per_level_scale}")

        self.base_resolution = base_resolution
        print(f"Base resolution (base_resolution): {self.base_resolution}")

        self.log2_hashmap_size = log2_hashmap_size
        print(f"log2 Hashmap size (log2_hashmap_size): {self.log2_hashmap_size}")

        # Compute features per level
        self.F = dim // num_levels  # Features per level
        assert self.F * self.num_levels == dim, "dim must be divisible by num_levels"

        # Create hash tables as learnable parameters for each level
        self.hash_tables = nn.ModuleList([
            nn.Embedding(2 ** self.log2_hashmap_size, self.F)
            for _ in range(self.num_levels)
        ])

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape [B, 3, N], representing normalized coordinates in [0, 1].

        Returns:
            embeddings: Tensor of shape [B, dim, N], multi-resolution hash embeddings.
        """
        B, _, N = coords.shape
        device = coords.device

        # Normalize coordinates to [0, 1]
        coords_min = coords.amin(dim=2, keepdim=True)
        coords_max = coords.amax(dim=2, keepdim=True)
        coords_range = coords_max - coords_min + 1e-6
        coords_norm = (coords - coords_min) / coords_range  # [B, 3, N]
        coords_norm = coords_norm.permute(0, 2, 1).contiguous()  # [B, N, 3]

        embeddings = []

        for level in range(self.num_levels):
            resolution = self.base_resolution * (self.per_level_scale ** level)
            coords_scaled = coords_norm * resolution  # Scale coordinates

            # Get integer and fractional parts
            coords_floor = torch.floor(coords_scaled).long()  # [B, N, 3]
            coords_frac = coords_scaled - coords_floor.float()  # [B, N, 3]

            # Compute the 8 corner indices for trilinear interpolation
            x0 = coords_floor[..., 0]
            y0 = coords_floor[..., 1]
            z0 = coords_floor[..., 2]

            x1 = x0 + 1
            y1 = y0 + 1
            z1 = z0 + 1

            # Wrap around for hash table indices
            def hash_coords(x, y, z):
                # Simple spatial hash
                primes = torch.tensor([1, 2654435761, 805459861], device=device, dtype=torch.long)
                return (x * primes[0] + y * primes[1] + z * primes[2]) & (2 ** self.log2_hashmap_size - 1)

            h000 = hash_coords(x0, y0, z0)
            h001 = hash_coords(x0, y0, z1)
            h010 = hash_coords(x0, y1, z0)
            h011 = hash_coords(x0, y1, z1)
            h100 = hash_coords(x1, y0, z0)
            h101 = hash_coords(x1, y0, z1)
            h110 = hash_coords(x1, y1, z0)
            h111 = hash_coords(x1, y1, z1)

            # Retrieve embeddings from hash tables
            e000 = self.hash_tables[level](h000)  # [B, N, F]
            e001 = self.hash_tables[level](h001)
            e010 = self.hash_tables[level](h010)
            e011 = self.hash_tables[level](h011)
            e100 = self.hash_tables[level](h100)
            e101 = self.hash_tables[level](h101)
            e110 = self.hash_tables[level](h110)
            e111 = self.hash_tables[level](h111)

            # Compute trilinear weights
            wx = coords_frac[..., 0].unsqueeze(-1)  # [B, N, 1]
            wy = coords_frac[..., 1].unsqueeze(-1)
            wz = coords_frac[..., 2].unsqueeze(-1)

            w000 = (1 - wx) * (1 - wy) * (1 - wz)
            w001 = (1 - wx) * (1 - wy) * wz
            w010 = (1 - wx) * wy * (1 - wz)
            w011 = (1 - wx) * wy * wz
            w100 = wx * (1 - wy) * (1 - wz)
            w101 = wx * (1 - wy) * wz
            w110 = wx * wy * (1 - wz)
            w111 = wx * wy * wz

            # Weighted sum of embeddings
            embedding = (
                e000 * w000 +
                e001 * w001 +
                e010 * w010 +
                e011 * w011 +
                e100 * w100 +
                e101 * w101 +
                e110 * w110 +
                e111 * w111
            )  # [B, N, F]

            embeddings.append(embedding)

        # Concatenate embeddings from all levels
        embeddings = torch.cat(embeddings, dim=-1)  # [B, N, dim]
        embeddings = embeddings.permute(0, 2, 1).contiguous()  # [B, dim, N]

        return embeddings

class BasicPositionalEncoding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Conv1d(3, dim, kernel_size=1, bias=True),
            nn.GroupNorm(8, dim),
            Swish(),
            nn.Conv1d(dim, dim, kernel_size=1, bias=True),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        # coords shape: [B, 3, N]
        pos_enc = self.proj(coords)  # [B, dim, N]
        return pos_enc
class FrequencyEncoding(nn.Module):
    def __init__(self, dim: int):
        """
        Standard sinusoidal positional encoding with scaled L2 norm.
        Args:
            dim: Desired output dimension (should be 64 for this implementation).
        """
        super().__init__()
        print("Using standard sinusoidal positional encoding with scaled L2 norm.")

        self.dim = dim  # Desired output dimension (e.g., 64)
        print(f"Output dimension (dim): {self.dim}")

        # Compute the number of frequency bands L
        # The total dimension D is given by D = 4 + 6L
        # Solve for L: L = (dim - 4) // 6
        assert (self.dim - 4) % 6 == 0, "dim - 4 must be divisible by 6"
        self.L = (self.dim - 4) // 6
        print(f"Number of frequency bands (L): {self.L}")

        # Frequencies: [2^0, 2^1, ..., 2^{L-1}]
        self.freq_bands = 2.0 ** torch.linspace(0, self.L - 1, self.L)
        print(f"Frequency bands: {self.freq_bands.tolist()}")

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Tensor of shape [B, 3, N], representing coordinates.

        Returns:
            embeddings: Tensor of shape [B, dim, N], positional encodings.
        """
        B, C, N = coords.shape
        assert C == 3, "Input coordinates should have 3 channels (x, y, z)"
        device = coords.device

        # Normalize coordinates to [0, 1]
        coords_min = coords.amin(dim=2, keepdim=True)
        coords_max = coords.amax(dim=2, keepdim=True)
        coords_range = coords_max - coords_min + 1e-6  # Avoid division by zero
        coords_norm = (coords - coords_min) / coords_range  # [B, 3, N]

        # Compute scaled L2 norm of coordinates
        l2_norm = torch.norm(coords_norm, dim=1, keepdim=True)  # [B, 1, N]
        # Scale L2 norm to [0, 1]
        l2_norm_min = l2_norm.amin(dim=2, keepdim=True)
        l2_norm_max = l2_norm.amax(dim=2, keepdim=True)
        l2_norm_range = l2_norm_max - l2_norm_min + 1e-6
        l2_norm_scaled = (l2_norm - l2_norm_min) / l2_norm_range  # [B, 1, N]

        # Prepare frequency bands
        freq_bands = self.freq_bands.to(device)  # [L]

        # Compute sinusoidal encodings for each coordinate
        embeddings = [coords_norm]  # Start with normalized coordinates [B, 3, N]

        for freq in freq_bands:
            for fn in [torch.sin, torch.cos]:
                embeddings.append(fn(coords_norm * freq * np.pi))  # [B, 3, N]

        # Concatenate all embeddings
        embeddings = torch.cat(embeddings, dim=1)  # [B, 3 + 2*L*3, N]

        # Append scaled L2 norm
        embeddings = torch.cat([embeddings, l2_norm_scaled], dim=1)  # [B, D, N], D = 4 + 6L

        # Ensure the output dimension matches self.dim
        assert embeddings.shape[1] == self.dim, f"Output dimension {embeddings.shape[1]} does not match expected {self.dim}"

        return embeddings  # [B, dim, N]
