from .common import xp
from .multi_head_attention import MultiHeadAttention
from .position_wise_ffn import PositionWiseFeedForwardNetwork
from .residual_layer_norm import ResidualLayerNorm

class DecoderLayer:
    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1):
        self.self_attn  = MultiHeadAttention(d_model, num_heads)
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.ffn        = PositionWiseFeedForwardNetwork(d_model, d_ff)
        self.norm1, self.norm2, self.norm3 = [ResidualLayerNorm(d_model) for _ in range(3)]
        self.dropout    = dropout

    def get_parameters_dict(self) -> dict:
        params = {}
        for name, p in self.self_attn.get_parameters_dict().items(): params[f"self_attn.{name}"] = p
        for name, p in self.cross_attn.get_parameters_dict().items(): params[f"cross_attn.{name}"] = p
        for name, p in self.ffn.get_parameters_dict().items(): params[f"ffn.{name}"] = p
        for name, p in self.norm1.get_parameters_dict().items(): params[f"norm1.{name}"] = p
        for name, p in self.norm2.get_parameters_dict().items(): params[f"norm2.{name}"] = p
        for name, p in self.norm3.get_parameters_dict().items(): params[f"norm3.{name}"] = p
        return params


    def set_parameters_dict(self, params: dict):
        """Delega o carregamento de pesos para as sub-camadas."""
        sa_params, ca_params, ffn_params = {}, {}, {}
        n1_params, n2_params, n3_params = {}, {}, {}
        
        for key, value in params.items():
            if key.startswith("self_attn."):
                sa_params[key.replace("self_attn.", "")] = value
            elif key.startswith("cross_attn."):
                ca_params[key.replace("cross_attn.", "")] = value
            elif key.startswith("ffn."):
                ffn_params[key.replace("ffn.", "")] = value
            elif key.startswith("norm1."):
                n1_params[key.replace("norm1.", "")] = value
            elif key.startswith("norm2."):
                n2_params[key.replace("norm2.", "")] = value
            elif key.startswith("norm3."):
                n3_params[key.replace("norm3.", "")] = value
        
        self.self_attn.set_parameters_dict(sa_params)
        self.cross_attn.set_parameters_dict(ca_params)
        self.ffn.set_parameters_dict(ffn_params)
        self.norm1.set_parameters_dict(n1_params)
        self.norm2.set_parameters_dict(n2_params)
        self.norm3.set_parameters_dict(n3_params)

    def forward(self, x: xp.ndarray, memory: xp.ndarray, tgt_mask: xp.ndarray | None = None, src_mask: xp.ndarray | None = None):
        attn1 = self.self_attn.forward(x, x, x, mask=tgt_mask)
        x1    = self.norm1.forward(x, attn1)
        
        attn2 = self.cross_attn.forward(x1, memory, memory, mask=src_mask)
        x2    = self.norm2.forward(x1, attn2)
        
        ffn_out = self.ffn.forward(x2)
        return self.norm3.forward(x2, ffn_out)

    def backward(self, d_output):
        # Ordem inversa
        dx2_norm3, dffn_out, norm3_grads = self.norm3.backward(d_output)
        dx2_ffn, ffn_grads = self.ffn.backward(dffn_out)
        dx2 = dx2_norm3 + dx2_ffn
        
        dx1_norm2, dattn2, norm2_grads = self.norm2.backward(dx2)
        dx1_cross, dmem, _, cross_attn_grads = self.cross_attn.backward(dattn2)
        dx1 = dx1_norm2 + dx1_cross
        
        dx_norm1, dattn1, norm1_grads = self.norm1.backward(dx1)
        dq, dk, dv, self_attn_grads = self.self_attn.backward(dattn1)
        dx = dx_norm1 + dq + dk + dv
        
        # Junta os gradientes
        grads = {}
        for name, grad in self_attn_grads.items(): grads[f"self_attn.{name}"] = grad
        for name, grad in cross_attn_grads.items(): grads[f"cross_attn.{name}"] = grad
        for name, grad in ffn_grads.items(): grads[f"ffn.{name}"] = grad
        for name, grad in norm1_grads.items(): grads[f"norm1.{name}"] = grad
        for name, grad in norm2_grads.items(): grads[f"norm2.{name}"] = grad
        for name, grad in norm3_grads.items(): grads[f"norm3.{name}"] = grad
        
        return dx, dmem, grads