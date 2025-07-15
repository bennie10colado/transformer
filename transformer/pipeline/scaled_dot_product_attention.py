from .common import xp
from .softmax import softmax
import math

def scaled_dot_product_attention(query, key, value, mask=None):
    """
    Calcula a atenção escalada.
    
    <--- MUDANÇA: Agora retorna a saída e os pesos da atenção.
    """
    dk = query.shape[-1]
    
    # 1. Scores = (Q @ K.T) / sqrt(dk)
    scores = (query @ key.swapaxes(-2, -1)) / math.sqrt(dk)

    # 2. Aplica a máscara (se houver)
    if mask is not None:
        scores += mask

    # 3. Pesos = softmax(scores)
    weights = softmax(scores)

    # 4. Saída = pesos @ V
    output = weights @ value
    
    # Retorna tanto a saída quanto os pesos (necessários para o backward)
    return output, weights

def scaled_dot_product_attention_backward(d_output, query, key, value, weights):
    """
    Calcula os gradientes para a atenção escalada.
    
    Args:
        d_output: Gradiente vindo da camada seguinte (dL/dOutput).
        query, key, value: Entradas originais para a camada.
        weights: Pesos da atenção calculados no forward pass.

    Returns:
        Gradientes em relação a query, key e value (dq, dk, dv).
    """
    dk = query.shape[-1]

    # --- Backpropagation (ordem inversa do forward) ---
    
    # Forward 4: output = weights @ value
    # Gradientes para 'weights' e 'value'
    d_weights = d_output @ value.swapaxes(-2, -1)
    d_value = weights.swapaxes(-2, -1) @ d_output
    
    # Forward 3: weights = softmax(scores)
    # Gradiente para 'scores'
    d_scores = softmax(weights, d_weights)

    # Forward 2: scores = scores + mask (o gradiente passa direto, pois mask é constante)
    
    # Forward 1: scores = (Q @ K.T) / sqrt(dk)
    # Gradiente para 'scores' antes da escala
    d_scores_scaled = d_scores / math.sqrt(dk)
    
    # Gradientes para 'query' e 'key'
    d_query = d_scores_scaled @ key
    d_key = d_scores_scaled.swapaxes(-2, -1) @ query
    
    return d_query, d_key, d_value