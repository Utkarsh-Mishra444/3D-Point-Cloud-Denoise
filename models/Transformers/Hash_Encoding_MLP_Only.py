from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import models.train_utils as train_utils
from models.modules import Attention
from models.Transformers.Input_Encodings import HashEncoding
from ..pvcnn import (
    PVCData,
    SharedMLP,
    Swish,
    create_mlp_components,
)

class HashMLP(nn.Module):
    def __init__(
        self,
        cfg: Dict,
        return_layers: bool = False,
    ):
        super().__init__()

        model_cfg = cfg.model
        pvd_cfg = model_cfg.PVD

        # Initialize class variables
        self.return_layers = return_layers
        self.input_dim = train_utils.default(model_cfg.in_dim, 3)
        self.extra_feature_channels = model_cfg.get("extra_feature_channels", 0)
        self.embed_dim = train_utils.default(model_cfg.time_embed_dim, 64)

        out_dim = train_utils.default(model_cfg.out_dim, 3)
        dropout = train_utils.default(model_cfg.dropout, 0.1)

        self.embedf = nn.Sequential(
            nn.Linear(self.embed_dim, self.embed_dim),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Linear(self.embed_dim, self.embed_dim),
        )

        self.f_embed_dim = pvd_cfg.get("feat_embed_dim", self.extra_feature_channels)

        # Feature embedding
        if self.f_embed_dim != self.extra_feature_channels or self.extra_feature_channels == 0:
            in_dim = self.extra_feature_channels if self.extra_feature_channels > 0 else self.input_dim
            self.embed_feats = nn.Sequential(
                nn.Conv1d(in_dim, self.f_embed_dim, kernel_size=1, bias=True),
                nn.GroupNorm(8, self.f_embed_dim),
                Swish(),
                nn.Conv1d(self.f_embed_dim, self.f_embed_dim, kernel_size=1, bias=True),
            )
        else:
            self.embed_feats = None

        # Positional Encoding
        self.pos_enc = HashEncoding(self.f_embed_dim)

        # Output projection (4-layer MLP)
        in_channels = self.f_embed_dim + self.embed_dim  # Combined feature and time embedding dimensions
        mlp_layer_sizes = [64, 64, 64]  # Four hidden layers of size 256
        out_dim = train_utils.default(model_cfg.out_dim, 3)
        dropout = train_utils.default(model_cfg.dropout, 0.1)
        print(mlp_layer_sizes)
        # Construct the MLP layers
        layers = []
        prev_channels = in_channels
        for size in mlp_layer_sizes:
            layers.append(nn.Conv1d(prev_channels, size, kernel_size=1, bias=True))
            layers.append(nn.GroupNorm(8, size))
            layers.append(Swish())
            layers.append(nn.Dropout(dropout))
            prev_channels = size
        # Final output layer
        layers.append(nn.Conv1d(prev_channels, out_dim, kernel_size=1, bias=True))
        self.classifier = nn.Sequential(*layers)

    def get_timestep_embedding(self, timesteps, device):
        if len(timesteps.shape) == 2 and timesteps.shape[1] == 1:
            timesteps = timesteps[:, 0]
        assert len(timesteps.shape) == 1, f"get shape: {timesteps.shape}"

        half_dim = self.embed_dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.from_numpy(np.exp(np.arange(0, half_dim) * -emb)).float().to(device)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if self.embed_dim % 2 == 1:  # zero pad
            emb = nn.functional.pad(emb, (0, 1), "constant", 0)
        assert emb.shape == torch.Size([timesteps.shape[0], self.embed_dim])
        return emb

    def forward(self, x, t, x_cond=None):
        if x_cond is not None:
            x = torch.cat([x, x_cond], dim=1)

        (B, C, N), device = x.shape, x.device
        assert (
            C == self.input_dim + self.extra_feature_channels
        ), f"input dim: {C}, expected: {self.input_dim + self.extra_feature_channels}"

        coords = x[:, : self.input_dim, :].contiguous()  # [B, 3, N]
        features = x[:, self.input_dim :, :].contiguous()  # [B, C_feat, N]

        # Embed features
        if self.embed_feats is not None:
            if self.extra_feature_channels == 0:
                features = self.embed_feats(coords)
            else:
                features = self.embed_feats(features)

        # Positional Encoding
        pos_enc = self.pos_enc(coords)  # [B, dim, N]

        # Combine features and positional encoding
        features = features + pos_enc  # [B, dim, N]

        # Time embedding
        if t is not None:
            if t.ndim == 0 and not len(t.shape) == 1:
                t = t.view(1).expand(B)
            time_emb = self.embedf(self.get_timestep_embedding(t, device))  # [B, embed_dim]
            time_emb = time_emb.unsqueeze(-1).expand(-1, -1, N)  # [B, embed_dim, N]
            features = torch.cat([features, time_emb], dim=1)  # [B, dim + embed_dim, N]

        # Output projection (pass through MLP)
        output = self.classifier(features)  # [B, out_dim, N]

        return output  # [B, out_dim, N]
