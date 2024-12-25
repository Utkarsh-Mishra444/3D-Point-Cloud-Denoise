
### Transformer with decoder

from functools import partial
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

import models.train_utils as train_utils

# New imports for transformer layers
from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoder, TransformerDecoderLayer
from models.Transformers.Input_Encodings import HashEncoding

class Full_Transformer(nn.Module):
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
        self.extra_feature_channels = 0  # Removed extra features

        self.embed_dim = train_utils.default(model_cfg.time_embed_dim, 128)

        out_dim = train_utils.default(model_cfg.out_dim, 3)
        dropout = train_utils.default(model_cfg.dropout, 0.1)
        num_heads = train_utils.default(pvd_cfg.get("transformer_heads", 8), 8)
        num_encoder_layers = train_utils.default(pvd_cfg.get("transformer_layers", 6), 6)
        num_decoder_layers = train_utils.default(pvd_cfg.get("transformer_decoder_layers", 6), 6)
        transformer_ffn_dim = train_utils.default(pvd_cfg.get("transformer_ffn_dim", 512), 512)

        # Positional Encoding
        self.pos_enc = HashEncoding(dim=64)  # Set dim to 128

        # Feature embedding dimension
        self.f_embed_dim = self.pos_enc.dim  # 128

        # Time embedding dimension
        self.embed_dim = self.f_embed_dim  # 128

        # self.embedf = nn.Sequential(
        #     nn.Linear(self.embed_dim, self.embed_dim),
        #     nn.LeakyReLU(0.1, inplace=True),
        #     nn.Linear(self.embed_dim, self.embed_dim),
        # )

        # Transformer dimensions
        transformer_d_model = self.f_embed_dim  # 128

        # Transformer Encoder Layers
        encoder_layer = TransformerEncoderLayer(
            d_model=transformer_d_model,
            nhead=num_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,  # Enable batch first
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Coordinate embedding for decoder input
        self.coord_embed = nn.Sequential(
            nn.Linear(self.input_dim, transformer_d_model),
            nn.Dropout(dropout),  # Add dropout
            nn.ReLU(),
        )

        # Transformer Decoder Layers
        decoder_layer = TransformerDecoderLayer(
            d_model=transformer_d_model,
            nhead=num_heads,
            dim_feedforward=transformer_ffn_dim,
            dropout=dropout,
            activation='relu',
            batch_first=True,
        )
        self.transformer_decoder = TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(transformer_d_model, transformer_d_model),
            nn.Dropout(dropout),  # Add dropout
            nn.ReLU(),
            nn.Linear(transformer_d_model, out_dim),
        )

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
            C == self.input_dim
        ), f"input dim: {C}, expected: {self.input_dim}"

        coords = x[:, : self.input_dim, :].contiguous()  # [B, 3, N]

        # Positional Encoding
        features = self.pos_enc(coords)  # [B, dim, N]

        # Time embedding
        if t is not None:
            if t.ndim == 0 and not len(t.shape) == 1:
                t = t.view(1).expand(B)
            time_emb = self.get_timestep_embedding(t, device) #self.embedf(self.get_timestep_embedding(t, device))  # [B, embed_dim]
            time_emb = time_emb.unsqueeze(-1).expand(-1, -1, N)  # [B, embed_dim, N]
            features = features + time_emb  # Add time embedding

        # Prepare features for transformer
        features = features.permute(0, 2, 1)  # [B, N, C] where C = dim = 128

        # Transformer Encoder
        memory = self.transformer_encoder(features)  # [B, N, C]

        # # Prepare tgt for decoder
        # tgt = coords.permute(0, 2, 1).contiguous()  # [B, N, input_dim]
        # tgt = self.coord_embed(tgt)  # [B, N, transformer_d_model]

        tgt = features
        
        # Transformer Decoder
        output = self.transformer_decoder(tgt, memory=memory)  # [B, N, transformer_d_model]

        # Output projection
        output = self.output_proj(output)  # [B, N, out_dim]
        output = output.permute(0, 2, 1).contiguous()  # [B, out_dim, N]

        return output  # [B, out_dim, N]
