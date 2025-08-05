from .common import xp
from .token_embedding import create_embedding, embed_tokens
from .positional_encoding import get_positional_encoding
from .encoder_layer import EncoderLayer
from .decoder_layer import DecoderLayer
from .loss import label_smoothing_grad
import math

class Transformer:
    def __init__(self,
                 vocab_size: int,
                 d_model: int,
                 n_layers: int,
                 n_heads: int,
                 d_ff: int,
                 max_len: int,
                 dropout: float = 0.1):  # Dropout como parâmetro do modelo

        # --- parâmetros aprendíveis ---
        self.Wemb_src = create_embedding(vocab_size, d_model)             # (V, D)
        self.Wemb_tgt = create_embedding(vocab_size, d_model)             # (V, D)
        self.pe       = get_positional_encoding(max_len, d_model)         # (1, L, D)
        self.enc = [EncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        self.dec = [DecoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]

        # --- MUDANÇA 1: WEIGHT SHARING ---
        # Wout não é mais um parâmetro separado. Ele é uma referência direta
        # para a matriz de embedding do target.
        self.Wout = self.Wemb_tgt

        self._cache = {}

    def forward(self,
                src_ids:  xp.ndarray,                  # (B, Lsrc)
                tgt_ids:  xp.ndarray,                  # (B, Ltgt)
                src_mask: xp.ndarray | None = None,    # (B, Lsrc, 1) ou None
                tgt_mask: xp.ndarray | None = None,    # (B, Ltgt, Ltgt) ou None
                training: bool = True                  # se True, aplica dropout
               ) -> xp.ndarray:                        # (B, Ltgt, V)
        B, Ls = src_ids.shape
        _, Lt = tgt_ids.shape

        # 1) Embeddings + Positional Encoding
        Es = embed_tokens(src_ids, self.Wemb_src) * math.sqrt(self.Wemb_src.shape[1])
        Es = Es + self.pe[:, :Ls, :]
        Et = embed_tokens(tgt_ids, self.Wemb_tgt) * math.sqrt(self.Wemb_tgt.shape[1])
        Et = Et + self.pe[:, :Lt, :]

        # armazena valores iniciais no cache para o backward
        self._cache['src_ids'] = src_ids
        self._cache['tgt_ids'] = tgt_ids
        self._cache['Es'] = Es
        self._cache['Et'] = Et

        # 2) Pilha do Encoder
        xs = Es
        for i, layer in enumerate(self.enc):
            # CORREÇÃO: Passa o flag 'training' e armazena o cache da camada
            xs, layer_cache = layer.forward(xs, src_mask, training=training)
            self._cache[f'enc_layer_{i}_cache'] = layer_cache
        memory = xs

        # 3) Pilha do Decoder
        xt = Et
        for i, layer in enumerate(self.dec):
            # CORREÇÃO: Passa 'memory', o flag 'training' e armazena o cache da camada
            xt, layer_cache = layer.forward(xt, memory, tgt_mask, src_mask, training=training)
            self._cache[f'dec_layer_{i}_cache'] = layer_cache

        # 4) Projeção Final
        self._cache['xt_final'] = xt
        flat = xt.reshape(-1, xt.shape[-1])

        # --- MUDANÇA 2: PROJEÇÃO FINAL COM PESOS COMPARTILHADOS ---
        # Usamos a transposta da matriz de embedding para a projeção final.
        # Shape: (B*Lt, D) @ (D, V) -> (B*Lt, V)
        logits = flat @ self.Wout.T
        # --- FIM DA MUDANÇA ---

        return logits.reshape(B, Lt, -1)

    def get_parameters_dict(self) -> dict[str, xp.ndarray]:
        # --- MUDANÇA 3: REMOVER Wout DOS PARÂMETROS ---
        # Wout não é mais um parâmetro independente, então o removemos da lista.
        params = {
            "Wemb_src": self.Wemb_src,
            "Wemb_tgt": self.Wemb_tgt,
        }
        # --- FIM DA MUDANÇA ---

        # Coleta parâmetros de todas as camadas do Encoder
        for i, layer in enumerate(self.enc):
            for p_name, p_val in layer.self_attn.get_parameters().items():
                params[f"enc_{i}_sa_{p_name}"] = p_val
            for p_name, p_val in layer.ffn.get_parameters().items():
                params[f"enc_{i}_ffn_{p_name}"] = p_val
            params[f"enc_{i}_norm1_gamma"] = layer.norm1.gamma
            params[f"enc_{i}_norm1_beta"] = layer.norm1.beta
            params[f"enc_{i}_norm2_gamma"] = layer.norm2.gamma
            params[f"enc_{i}_norm2_beta"] = layer.norm2.beta

        # Coleta parâmetros de todas as camadas do Decoder
        for i, layer in enumerate(self.dec):
            for p_name, p_val in layer.self_attn.get_parameters().items():
                params[f"dec_{i}_sa_{p_name}"] = p_val
            for p_name, p_val in layer.cross_attn.get_parameters().items():
                params[f"dec_{i}_ca_{p_name}"] = p_val
            for p_name, p_val in layer.ffn.get_parameters().items():
                params[f"dec_{i}_ffn_{p_name}"] = p_val
            params[f"dec_{i}_norm1_gamma"] = layer.norm1.gamma
            params[f"dec_{i}_norm1_beta"] = layer.norm1.beta
            params[f"dec_{i}_norm2_gamma"] = layer.norm2.gamma
            params[f"dec_{i}_norm2_beta"] = layer.norm2.beta
            params[f"dec_{i}_norm3_gamma"] = layer.norm3.gamma
            params[f"dec_{i}_norm3_beta"] = layer.norm3.beta

        return params

    def backward(self,
                 logits: xp.ndarray,
                 tgt_out: xp.ndarray,
                 pad_idx: int,
                 epsilon: float = 0.1) -> list[xp.ndarray]:
        
        params_dict = self.get_parameters_dict()
        params_keys = list(params_dict.keys())
        grads = {name: xp.zeros_like(val) for name, val in params_dict.items()}

        dlogits = label_smoothing_grad(logits, tgt_out, pad_idx, epsilon)
        xt_final = self._cache['xt_final']
        B, Lt, D = xt_final.shape
        V = self.Wemb_tgt.shape[0]
        flat_xt = xt_final.reshape(-1, D)
        flat_dlog = dlogits.reshape(-1, V)

        # --- MUDANÇA 4: GRADIENTES COMPARTILHADOS ---
        # O gradiente da projeção final agora é acumulado em Wemb_tgt.
        # 1. Calcula o gradiente para Wout.T (shape D, V)
        grad_Wout_T = flat_xt.T @ flat_dlog
        # 2. Acumula este gradiente em Wemb_tgt (transpondo de volta para shape V, D)
        grads["Wemb_tgt"] += grad_Wout_T.T
        # 3. O gradiente dxt é calculado com Wout (que é Wemb_tgt, shape V, D)
        dxt = (flat_dlog @ self.Wout).reshape(B, Lt, D)
        # --- FIM DA MUDANÇA ---

        d_memory_total = xp.zeros_like(self._cache['Es'])
        for i in reversed(range(len(self.dec))):
            layer_cache = self._cache[f'dec_layer_{i}_cache']
            dxt, d_memory_from_dec, dec_layer_grads = self.dec[i].backward(dxt, layer_cache)
            d_memory_total += d_memory_from_dec
            for name, grad in dec_layer_grads.items():
                # Esta lógica assume que os gradientes de dec_layer_grads já vêm
                # com prefixos como 'sa_', 'ca_', etc.
                grads[f"dec_{i}_{name}"] = grad


        # O gradiente do embedding do target também é acumulado em Wemb_tgt
        # 4. Retropropagação pelo Embedding do Target
        d_Et = dxt * math.sqrt(self.Wemb_tgt.shape[1])
        xp.add.at(grads["Wemb_tgt"], self._cache['tgt_ids'].reshape(-1), d_Et.reshape(-1, D))

        # 5. Retropropagação pelo Encoder
        dxs = d_memory_total
        for i in reversed(range(len(self.enc))):
            layer_cache = self._cache[f'enc_layer_{i}_cache']
            dxs, enc_layer_grads = self.enc[i].backward(dxs, layer_cache)
            for name, grad in enc_layer_grads.items():
                grads[f"enc_{i}_{name}"] = grad

        d_Es = dxs * math.sqrt(self.Wemb_src.shape[1])
        xp.add.at(grads["Wemb_src"], self._cache['src_ids'].reshape(-1), d_Es.reshape(-1, D))

        return [grads[name] for name in params_keys]