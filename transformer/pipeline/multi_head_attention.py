from .common import xavier_uniform_init, xp
import math
from .scaled_dot_product_attention import scaled_dot_product_attention, scaled_dot_product_attention_backward

class MultiHeadAttention:
    def __init__(self, embed_dim: int, num_heads: int):
        assert embed_dim % num_heads == 0, "A dimensão do embedding deve ser divisível pelo número de cabeças."
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        limit = math.sqrt(6 / (embed_dim + embed_dim))
        self.W_q = xavier_uniform_init((embed_dim, embed_dim))
        self.W_k = xavier_uniform_init((embed_dim, embed_dim))
        self.W_v = xavier_uniform_init((embed_dim, embed_dim))
        self.W_o = xavier_uniform_init((embed_dim, embed_dim))

    def get_parameters(self) -> dict[str, xp.ndarray]:
        return {'W_q': self.W_q, 'W_k': self.W_k, 'W_v': self.W_v, 'W_o': self.W_o}

    def _split_heads(self, x: xp.ndarray) -> xp.ndarray:
        B, T, _ = x.shape
        x = x.reshape(B, T, self.num_heads, self.head_dim)
        return x.transpose(0, 2, 1, 3)

    def _combine_heads(self, x: xp.ndarray) -> xp.ndarray:
        B, nh, T, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, nh * dh)

    def forward(self, query: xp.ndarray, key: xp.ndarray, value: xp.ndarray, mask: xp.ndarray | None = None):
        q = query @ self.W_q
        k = key @ self.W_k
        v = value @ self.W_v
        q_h, k_h, v_h = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        context, attn_cache = scaled_dot_product_attention(q_h, k_h, v_h, mask)
        concat = self._combine_heads(context)
        output = concat @ self.W_o
        cache = (query, key, value, concat, attn_cache)
        return output, cache

    def backward(self, d_output, cache):
        query, key, value, concat, attn_cache = cache
        
        # --- CORREÇÃO AQUI ---
        # Achata os tensores 3D para 2D para a multiplicação de matrizes
        B, T, D = d_output.shape
        flat_concat = concat.reshape(-1, D)
        flat_d_output = d_output.reshape(-1, D)
        grad_Wo = flat_concat.T @ flat_d_output
        d_concat = (flat_d_output @ self.W_o.T).reshape(B, T, D)
        # --- FIM DA CORREÇÃO ---

        d_context = self._split_heads(d_concat)
        dq_h, dk_h, dv_h = scaled_dot_product_attention_backward(d_context, attn_cache)
        dq, dk, dv = self._combine_heads(dq_h), self._combine_heads(dk_h), self._combine_heads(dv_h)
        
        # Achata as entradas e gradientes para calcular os gradientes dos pesos
        flat_query = query.reshape(-1, D)
        flat_key = key.reshape(-1, D)
        flat_value = value.reshape(-1, D)
        flat_dq = dq.reshape(-1, D)
        flat_dk = dk.reshape(-1, D)
        flat_dv = dv.reshape(-1, D)

        grad_Wq = flat_query.T @ flat_dq
        grad_Wk = flat_key.T @ flat_dk
        grad_Wv = flat_value.T @ flat_dv
        
        d_query = dq @ self.W_q.T
        d_key = dk @ self.W_k.T
        d_value = dv @ self.W_v.T
        
        grads = self.get_parameters()
        grads = {'W_q': grad_Wq, 'W_k': grad_Wk, 'W_v': grad_Wv, 'W_o': grad_Wo}
        
        return (d_query + d_key + d_value), grads
    
    def backward_cross(self, d_output, cache):
        query, key, value, concat, attn_cache = cache
        
        # --- CORREÇÃO AQUI (mesma lógica do backward) ---
        B, T, D = d_output.shape
        flat_concat = concat.reshape(-1, D)
        flat_d_output = d_output.reshape(-1, D)
        grad_Wo = flat_concat.T @ flat_d_output
        d_concat = (flat_d_output @ self.W_o.T).reshape(B, T, D)
        # --- FIM DA CORREÇÃO ---

        d_context = self._split_heads(d_concat)
        dq_h, dk_h, dv_h = scaled_dot_product_attention_backward(d_context, attn_cache)
        dq, dk, dv = self._combine_heads(dq_h), self._combine_heads(dk_h), self._combine_heads(dv_h)
        
        # Achata as entradas e gradientes para calcular os gradientes dos pesos
        flat_query = query.reshape(-1, D)
        flat_key = key.reshape(-1, D)
        flat_value = value.reshape(-1, D)
        flat_dq = dq.reshape(-1, D)
        flat_dk = dk.reshape(-1, D)
        flat_dv = dv.reshape(-1, D)

        grad_Wq = flat_query.T @ flat_dq
        grad_Wk = flat_key.T @ flat_dk
        grad_Wv = flat_value.T @ flat_dv

        d_query = dq @ self.W_q.T
        d_memory = (dk @ self.W_k.T) + (dv @ self.W_v.T)
        
        grads = self.get_parameters()
        grads = {'W_q': grad_Wq, 'W_k': grad_Wk, 'W_v': grad_Wv, 'W_o': grad_Wo}
        
        return d_query, d_memory, grads
