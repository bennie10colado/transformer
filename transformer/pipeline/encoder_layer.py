import math
from .common import xp
from .scaled_dot_product_attention import scaled_dot_product_attention, scaled_dot_product_attention_backward
from .position_wise_ffn import PositionWiseFeedForwardNetwork
from .residual_layer_norm import ResidualLayerNorm
from .multi_head_attention import MultiHeadAttention


class EncoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn       = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.norm1     = ResidualLayerNorm(d_model)
        self.norm2     = ResidualLayerNorm(d_model)
        self.dropout   = dropout

    def get_parameters_dict(self) -> dict:
        params = {}
        for name, p in self.self_attn.get_parameters_dict().items(): params[f"self_attn.{name}"] = p
        for name, p in self.ffn.get_parameters_dict().items(): params[f"ffn.{name}"] = p
        for name, p in self.norm1.get_parameters_dict().items(): params[f"norm1.{name}"] = p
        for name, p in self.norm2.get_parameters_dict().items(): params[f"norm2.{name}"] = p
        return params

    def set_parameters_dict(self, params: dict):
        """Delega o carregamento de pesos para as sub-camadas."""
        # Cria sub-dicionários para cada componente
        attn_params, ffn_params, norm1_params, norm2_params = {}, {}, {}, {}
        for key, value in params.items():
            if key.startswith("self_attn."):
                attn_params[key.replace("self_attn.", "")] = value
            elif key.startswith("ffn."):
                ffn_params[key.replace("ffn.", "")] = value
            elif key.startswith("norm1."):
                norm1_params[key.replace("norm1.", "")] = value
            elif key.startswith("norm2."):
                norm2_params[key.replace("norm2.", "")] = value
        
        self.self_attn.set_parameters_dict(attn_params)
        self.ffn.set_parameters_dict(ffn_params)
        self.norm1.set_parameters_dict(norm1_params)
        self.norm2.set_parameters_dict(norm2_params)

    def forward(self, x: xp.ndarray, mask: xp.ndarray | None = None):
        attn_out = self.self_attn.forward(x, x, x, mask=mask)
        x1       = self.norm1.forward(x, attn_out)
        ffn_out  = self.ffn.forward(x1)
        return self.norm2.forward(x1, ffn_out)

    def backward(self, d_output):
        # Ordem inversa do forward
        dx1_norm2, dffn_out, norm2_grads = self.norm2.backward(d_output)
        dx1_ffn, ffn_grads = self.ffn.backward(dffn_out)
        dx1 = dx1_norm2 + dx1_ffn
        
        dx_norm1, dattn_out, norm1_grads = self.norm1.backward(dx1)
        dq, dk, dv, attn_grads = self.self_attn.backward(dattn_out)
        dx = dx_norm1 + dq + dk + dv
        
        # Junta os gradientes
        grads = {}
        for name, grad in attn_grads.items(): grads[f"self_attn.{name}"] = grad
        for name, grad in ffn_grads.items(): grads[f"ffn.{name}"] = grad
        for name, grad in norm1_grads.items(): grads[f"norm1.{name}"] = grad
        for name, grad in norm2_grads.items(): grads[f"norm2.{name}"] = grad
        
        return dx, grads
