import torch
import torch.nn as nn

import tts.config as config


class TtsModel(nn.Module):
    def __init__(
        self,
        vocab_size: int = 2048,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(4096, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, src, tgt):
        device = src.device

        src_key_padding_mask = src == config.PAD_TOKEN_ID
        tgt_key_padding_mask = tgt == config.PAD_TOKEN_ID

        tgt_seq_len = tgt.shape[1]
        tgt_mask = torch.triu(
            torch.ones((tgt_seq_len, tgt_seq_len), device=device), diagonal=1
        ).bool()

        src_pos = torch.arange(src.shape[1], device=device).unsqueeze(0)
        src_emb = self.token_emb(src) + self.pos_emb(src_pos)

        tgt_pos = torch.arange(tgt.shape[1], device=device).unsqueeze(0)
        tgt_emb = self.token_emb(tgt) + self.pos_emb(tgt_pos)

        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )

        output = self.ln_f(output)
        logits = self.head(output)

        return logits
