import math

# ── Implementação PyTorch ───────────────────────────────────────────────
import torch
import torch.nn as nn

class TokenEmbedding(nn.Module):
    """
    Embedding layer em PyTorch, com Xavier-Uniform initialization.
    """
    def __init__(self, vocab_size: int, d_model: int, device: str = 'cpu'):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model    = d_model

        # Xavier-Uniform
        limit = math.sqrt(6 / (vocab_size + d_model))
        weights = (torch.rand(vocab_size, d_model, device=device) * 2 - 1) * limit
        self.embedding_weights = nn.Parameter(weights)

    def forward(self, token_ids: torch.LongTensor) -> torch.Tensor:
        """
        token_ids: LongTensor of shape (batch, seq_len)
        retorna:   Tensor of shape (batch, seq_len, d_model)
        """
        embedded = self.embedding_weights[token_ids]
        return embedded * math.sqrt(self.d_model)


# ── Implementação NumPy/CuPy ────────────────────────────────────────────
from .common import xavier_uniform_init, xp  # xp → numpy ou cupy

def create_embedding(vocab_size: int, d_model: int) -> xp.ndarray:
    """
    Cria matriz de embedding (vocab_size, d_model) Xavier-Uniform.
    """
    return xavier_uniform_init((vocab_size, d_model))


def embed_tokens(token_ids: xp.ndarray, W: xp.ndarray) -> xp.ndarray:
    """
    token_ids: array int32 (batch, seq_len)
    W:         array float32 (vocab_size, d_model)
    retorna:   array float32 (batch, seq_len, d_model)
    """
    B, T = token_ids.shape
    D    = W.shape[1]
    out  = xp.zeros((B, T, D), dtype=xp.float32)
    for i in range(B):
        for j in range(T):
            out[i, j] = W[token_ids[i, j]]
    return out * math.sqrt(D)
