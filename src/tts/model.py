# model.py
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

        # Embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # We use a learnable position embedding for simplicity in both Enc and Dec
        self.pos_emb = nn.Embedding(4096, d_model)

        # 1. THE ENCODER (Processes Phonemes)
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

        # 2. THE DECODER (Generates Audio, attending to Encoder)
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
        # src: [Batch, Src_Len] (Phonemes)
        # tgt: [Batch, Tgt_Len] (Audio Input)

        device = src.device

        # --- Masks ---
        # 1. Padding Mask for Source (Phonemes)
        src_key_padding_mask = src == config.PAD_TOKEN_ID

        # 2. Padding Mask for Target (Audio)
        tgt_key_padding_mask = tgt == config.PAD_TOKEN_ID

        # 3. Causal Mask for Target (Prevent looking ahead in audio)
        tgt_seq_len = tgt.shape[1]
        tgt_mask = torch.triu(
            torch.ones((tgt_seq_len, tgt_seq_len), device=device), diagonal=1
        ).bool()

        # --- Embedding + Positional Encoding ---
        # Encoder Source
        src_pos = torch.arange(src.shape[1], device=device).unsqueeze(0)
        src_emb = self.token_emb(src) + self.pos_emb(src_pos)

        # Decoder Target
        tgt_pos = torch.arange(tgt.shape[1], device=device).unsqueeze(0)
        tgt_emb = self.token_emb(tgt) + self.pos_emb(tgt_pos)

        # --- Transformer Pass ---

        # 1. Encode Phonemes
        # memory shape: [Batch, Src_Len, D_Model]
        memory = self.encoder(src_emb, src_key_padding_mask=src_key_padding_mask)

        # 2. Decode Audio (with Cross-Attention to Memory)
        output = self.decoder(
            tgt_emb,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,  # Mask padding in cross-attention
        )

        output = self.ln_f(output)
        logits = self.head(output)

        return logits
