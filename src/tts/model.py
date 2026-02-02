import torch
import torch.nn as nn

import tts.config as config


class BabyValle(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        max_len: int = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.type_emb = nn.Embedding(2, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x):
        B, T = x.shape
        device = x.device

        positions = torch.arange(T, device=device)

        token_types = (x >= 1024).long()

        x_emb = self.token_emb(x) + self.pos_emb(positions) + self.type_emb(token_types)

        attn_mask = nn.Transformer.generate_square_subsequent_mask(T).to(device)

        key_padding_mask = x == config.PAD_TOKEN_ID

        x = self.blocks(
            x_emb, mask=attn_mask, src_key_padding_mask=key_padding_mask, is_causal=True
        )

        x = self.ln_f(x)
        logits = self.head(x)

        return logits
