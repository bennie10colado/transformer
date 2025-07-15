from .common import xp

def create_embedding(vocab_size: int, d_model: int):
    """Cria uma matriz de embedding inicializada."""
    # Inicialização de Glorot/Xavier
    limit = xp.sqrt(6 / (vocab_size + d_model))
    return xp.random.uniform(-limit, limit, (vocab_size, d_model), dtype=xp.float32)

def embed_tokens(ids: xp.ndarray, W_emb: xp.ndarray) -> xp.ndarray:
    """
    Busca os vetores de embedding para uma sequência de IDs.
    Esta é a operação de "lookup" do forward pass.
    """
    return W_emb[ids]

def token_embedding_backward(d_embeddings: xp.ndarray, ids: xp.ndarray, W_emb: xp.ndarray) -> xp.ndarray:
    """
    Calcula o gradiente para a matriz de embedding (W_emb).
    Esta é a operação de "scatter-add" do backward pass.
    
    Args:
        d_embeddings: O gradiente vindo da camada seguinte (dL/dE).
                      Shape: (batch_size, seq_len, d_model)
        ids: Os IDs dos tokens originais.
             Shape: (batch_size, seq_len)
        W_emb: A matriz de embedding original, para sabermos seu shape.
    
    Returns:
        O gradiente para a matriz de embedding (dL/dW_emb).
        Shape: (vocab_size, d_model)
    """
    # Cria uma matriz de gradientes zerada com o mesmo shape da matriz de embedding
    grad_W_emb = xp.zeros_like(W_emb)
    
    # Achata os IDs e os gradientes para facilitar a indexação
    flat_ids = ids.flatten()
    flat_d_emb = d_embeddings.reshape(-1, d_embeddings.shape[-1])
    
    # A mágica acontece aqui: xp.add.at
    # Para cada ID em `flat_ids`, ele soma o gradiente correspondente
    # de `flat_d_emb` à linha apropriada em `grad_W_emb`.
    # É uma forma eficiente de fazer a operação de "espalhar e somar".
    xp.add.at(grad_W_emb, flat_ids, flat_d_emb)
    
    return grad_W_emb