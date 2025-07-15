import math
from .common import xp
from .scaled_dot_product_attention import scaled_dot_product_attention, scaled_dot_product_attention_backward

class MultiHeadAttention:
    """Implementa a camada de Multi-Head Attention."""
    
    def __init__(self, d_model: int, num_heads: int):
        assert d_model % num_heads == 0, "d_model deve ser divisível por num_heads"
        self.d_model, self.num_heads, self.dk = d_model, num_heads, d_model // num_heads
        
        limit = math.sqrt(6 / (d_model * 2))
        self.W_q = xp.random.uniform(-limit, limit, (d_model, d_model), dtype=xp.float32)
        self.W_k = xp.random.uniform(-limit, limit, (d_model, d_model), dtype=xp.float32)
        self.W_v = xp.random.uniform(-limit, limit, (d_model, d_model), dtype=xp.float32)
        self.W_o = xp.random.uniform(-limit, limit, (d_model, d_model), dtype=xp.float32)
        
        self._cache = {}

    def get_parameters_dict(self) -> dict:
        return {"W_q": self.W_q, "W_k": self.W_k, "W_v": self.W_v, "W_o": self.W_o}

    def set_parameters_dict(self, params: dict):
        for key in ["W_q", "W_k", "W_v", "W_o"]:
            setattr(self, key, xp.asarray(params[key]))

    def _split_heads(self, x: xp.ndarray):
        B, T, _ = x.shape
        return x.reshape(B, T, self.num_heads, self.dk).transpose(0, 2, 1, 3)

    def _combine_heads(self, x: xp.ndarray):
        B, _, T, _ = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, T, self.d_model)

    def forward(self, query: xp.ndarray, key: xp.ndarray, value: xp.ndarray, mask: xp.ndarray | None = None):
        self._cache['query'], self._cache['key'], self._cache['value'] = query, key, value
        
        q, k, v = query @ self.W_q, key @ self.W_k, value @ self.W_v
        
        q_split, k_split, v_split = self._split_heads(q), self._split_heads(k), self._split_heads(v)
        self._cache['q_split'], self._cache['k_split'], self._cache['v_split'] = q_split, k_split, v_split
        
        context, attn_weights = scaled_dot_product_attention(q_split, k_split, v_split, mask)
        self._cache['attn_weights'] = attn_weights

        concat = self._combine_heads(context)
        self._cache['concat'] = concat
        
        return concat @ self.W_o

    def backward(self, d_output):
        # 1. Gradiente da projeção de saída
        grad_Wo = self._cache['concat'].transpose(0, 2, 1).reshape(-1, self.d_model).T @ d_output.reshape(-1, self.d_model)
        d_concat = d_output @ self.W_o.T
        
        # 2. Retropropaga por combine_heads -> <<<<<<< CORREÇÃO AQUI >>>>>>>>>
        # Para reverter a junção, nós separamos as cabeças novamente.
        d_context = self._split_heads(d_concat)
        
        # 3. Retropropaga pela atenção escalada
        dq_s, dk_s, dv_s = scaled_dot_product_attention_backward(d_context, self._cache['q_split'], self._cache['k_split'], self._cache['v_split'], self._cache['attn_weights'])

        # 4. Retropropaga por split_heads
        dq = self._combine_heads(dq_s)
        dk = self._combine_heads(dk_s)
        dv = self._combine_heads(dv_s)
        
        # 5. Gradientes das projeções de entrada
        grad_Wq = self._cache['query'].transpose(0, 2, 1).reshape(-1, self.d_model).T @ dq.reshape(-1, self.d_model)
        grad_Wk = self._cache['key'].transpose(0, 2, 1).reshape(-1, self.d_model).T @ dk.reshape(-1, self.d_model)
        grad_Wv = self._cache['value'].transpose(0, 2, 1).reshape(-1, self.d_model).T @ dv.reshape(-1, self.d_model)

        # Gradientes em relação às entradas da camada
        d_query = dq @ self.W_q.T
        d_key = dk @ self.W_k.T
        d_value = dv @ self.W_v.T
        
        grads = {"W_q": grad_Wq, "W_k": grad_Wk, "W_v": grad_Wv, "W_o": grad_Wo}
        return d_query, d_key, d_value, grads