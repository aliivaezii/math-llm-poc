# Decoder-only Transformer model definition (PyTorch, CPU-only, no external ML libraries).

import torch
import torch.nn as nn


class TinyDecoderLM(nn.Module):
    """Decoder-only Transformer for character-level arithmetic.

    Uses nn.TransformerDecoder with a zero memory tensor so cross-attention
    is a no-op, giving a pure causal language model without an encoder.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 4,
        d_ff: int = 512,
        dropout: float = 0.1,
        max_length: int = 32,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_length, d_model)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.xavier_uniform_(module.weight)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) token ids → logits: (B, T, vocab_size)"""
        B, T = x.shape

        positions = torch.arange(T, device=x.device).unsqueeze(0)      # (1, T)
        h = self.token_emb(x) + self.pos_emb(positions)                # (B, T, d_model)

        # Upper-triangular True = "ignore this position" in PyTorch's convention.
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device), diagonal=1
        ).bool()

        # TransformerDecoder requires a memory tensor; zeros make cross-attention a no-op.
        memory = torch.zeros(B, 1, self.d_model, device=x.device)

        h = self.decoder(h, memory, tgt_mask=causal_mask)              # (B, T, d_model)
        h = self.norm(h)
        return self.head(h)                                             # (B, T, vocab_size)


if __name__ == "__main__":
    from src.tokenizer import Tokenizer

    tok = Tokenizer()
    model = TinyDecoderLM(vocab_size=tok.vocab_size)

    x = torch.randint(0, tok.vocab_size, (2, 16))
    logits = model(x)

    print(f"input  shape : {tuple(x.shape)}")
    print(f"output shape : {tuple(logits.shape)}")
    assert logits.shape == (2, 16, tok.vocab_size), "Unexpected output shape"

    total_params = sum(p.numel() for p in model.parameters())
    print(f"parameters   : {total_params:,}")
    print("Shape assertion OK")
