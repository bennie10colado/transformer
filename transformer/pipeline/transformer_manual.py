import math

from transformer.pipeline.softmax import softmax
from .common import xp
from .token_embedding import create_embedding, embed_tokens
from .positional_encoding import get_positional_encoding
from .encoder_layer import EncoderLayer
from .decoder_layer import DecoderLayer
from .loss import label_smoothing_grad

class Transformer:
    def __init__(self,
                 vocab_size: int,
                 d_model:    int,
                 n_layers:   int,
                 n_heads:    int,
                 d_ff:       int,
                 max_len:    int):
        
        self.Wemb_src = create_embedding(vocab_size, d_model)
        self.Wemb_tgt = create_embedding(vocab_size, d_model)
        self.pe       = get_positional_encoding(max_len, d_model)
        self.enc      = [EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.dec      = [DecoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)]
        self.Wout     = xp.random.randn(d_model, vocab_size, dtype=xp.float32) / math.sqrt(d_model)

        self._cache = {} # Usado para guardar valores para o backward

    def forward(self,
                src_ids:  xp.ndarray,
                tgt_ids:  xp.ndarray,
                src_mask: xp.ndarray | None = None,
                tgt_mask: xp.ndarray | None = None
               ) -> xp.ndarray:
        B, Ls = src_ids.shape
        _, Lt = tgt_ids.shape

        Es = embed_tokens(src_ids, self.Wemb_src) * math.sqrt(self.Wemb_src.shape[1])
        Es = Es + self.pe[:, :Ls, :]
        Et = embed_tokens(tgt_ids, self.Wemb_tgt) * math.sqrt(self.Wemb_tgt.shape[1])
        Et = Et + self.pe[:, :Lt, :]

        # Cache para o backward
        self._cache['Es'] = Es
        self._cache['Et'] = Et

        # Encoder
        xs = Es
        for i, layer in enumerate(self.enc):
            xs = layer.forward(xs, src_mask)
        
        # O resultado final do encoder é salvo no cache para o decoder usar
        self._cache['encoder_output'] = xs

        # Decoder
        xt = Et
        for i, layer in enumerate(self.dec):
            xt = layer.forward(xt, xs, tgt_mask, src_mask)

        # Projeção final
        self._cache['decoder_output'] = xt 
        flat = xt.reshape(-1, xt.shape[-1])
        logits = flat @ self.Wout
        return logits.reshape(B, Lt, -1)

    def get_parameters_dict(self) -> dict:
        """Coleta todos os parâmetros do modelo e de suas sub-camadas."""
        params = {"Wemb_src": self.Wemb_src, "Wemb_tgt": self.Wemb_tgt, "Wout": self.Wout}
        
        for i, layer in enumerate(self.enc):
            for name, param in layer.get_parameters_dict().items():
                params[f"enc.{i}.{name}"] = param
        
        for i, layer in enumerate(self.dec):
            for name, param in layer.get_parameters_dict().items():
                params[f"dec.{i}.{name}"] = param
                
        return params

    def set_parameters_dict(self, params: dict):
        """Carrega parâmetros para o modelo e suas sub-camadas."""
        for name in ["Wemb_src", "Wemb_tgt", "Wout"]:
            if name in params:
                setattr(self, name, xp.asarray(params[name]))

        encoder_layer_params = [{} for _ in self.enc]
        decoder_layer_params = [{} for _ in self.dec]

        for key, value in params.items():
            if key.startswith("enc."):
                parts = key.split('.')
                layer_idx = int(parts[1])
                param_name = ".".join(parts[2:])
                encoder_layer_params[layer_idx][param_name] = value
            elif key.startswith("dec."):
                parts = key.split('.')
                layer_idx = int(parts[1])
                param_name = ".".join(parts[2:])
                decoder_layer_params[layer_idx][param_name] = value

        for i, layer in enumerate(self.enc):
            if encoder_layer_params[i]:
                layer.set_parameters_dict(encoder_layer_params[i])
        
        for i, layer in enumerate(self.dec):
            if decoder_layer_params[i]:
                layer.set_parameters_dict(decoder_layer_params[i])

    def backward(self,
                 log_probs: xp.ndarray,
                 tgt_out:   xp.ndarray,
                 pad_idx:   int
                ) -> dict[str, xp.ndarray]:
        
        # 1. Gradiente inicial na saída
        B, Lt, V = log_probs.shape
        d_log_probs = xp.zeros_like(log_probs)
        # Cria o gradiente "one-hot" para a classe correta
        xp.put_along_axis(d_log_probs, tgt_out.reshape(B, Lt, 1), -1, axis=2)
        non_pad_mask = (tgt_out != pad_idx)
        # Normaliza o gradiente pelo número de tokens válidos
        d_log_probs /= non_pad_mask.sum()
        
        # Retropropaga pelo log-softmax para obter o gradiente dos logits
        dlogits = softmax(xp.exp(log_probs), d_log_probs)
        
        # 2. Gradientes da camada de projeção final
        xt = self._cache['decoder_output']
        D = xt.shape[-1]
        flat_xt = xt.reshape(-1, D)
        flat_dlog = dlogits.reshape(-1, V)
        
        grad_Wout = flat_xt.T @ flat_dlog
        dxt = flat_dlog @ self.Wout.T
        dxt = dxt.reshape(B, Lt, D)
        
        grads = {"Wout": grad_Wout}

        # 3. Retropropagação pelo Decoder
        d_memory_total = xp.zeros_like(self._cache['encoder_output'])
        for i in reversed(range(len(self.dec))):
            dxt, d_memory_from_dec, dec_grads = self.dec[i].backward(dxt)
            d_memory_total += d_memory_from_dec
            for name, grad in dec_grads.items():
                grads[f"dec.{i}.{name}"] = grad
        
        # --- CORREÇÃO AQUI: Backward para o embedding do target ---
        # dxt agora contém o gradiente para a entrada do decoder (Et)
        grad_Wemb_tgt = token_embedding_backward(dxt, self._cache['tgt_ids_input'], self.Wemb_tgt)
        grads["Wemb_tgt"] = grad_Wemb_tgt

        # 4. Retropropagação pelo Encoder
        d_encoder_output = d_memory_total
        for i in reversed(range(len(self.enc))):
            d_encoder_output, enc_grads = self.enc[i].backward(d_encoder_output)
            for name, grad in enc_grads.items():
                grads[f"enc.{i}.{name}"] = grad

        # --- CORREÇÃO AQUI: Backward para o embedding da source ---
        # d_encoder_output agora contém o gradiente para a entrada do encoder (Es)
        grad_Wemb_src = token_embedding_backward(d_encoder_output, self._cache['src_ids'], self.Wemb_src)
        grads["Wemb_src"] = grad_Wemb_src
        
        return grads