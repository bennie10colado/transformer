from .common import xavier_uniform_init, xp
import math

class PositionWiseFeedForwardNetwork:
    """
    Implementa a rede Feed-Forward (FFN) que é aplicada a cada posição independentemente.
    Isso consiste em duas transformações lineares com uma ativação ReLU no meio,
    [cite_start]conforme a seção 3.3 do artigo "Attention Is All You Need"[cite: 147, 151].
    """
    def __init__(self, d_model: int, d_ff: int):
        """
        Inicializa os pesos e biases das duas camadas lineares.

        Args:
            [cite_start]d_model (int): A dimensionalidade de entrada e saída da camada[cite: 151].
            [cite_start]d_ff (int): A dimensionalidade da camada interna (oculta)[cite: 151].
        """
        # Inicialização de pesos para a primeira camada linear (d_model -> d_ff)
        self.W1 = xavier_uniform_init((d_model, d_ff))
        self.b1 = xp.zeros((d_ff,), dtype=xp.float32)

        # Inicialização de pesos para a segunda camada linear (d_ff -> d_model)
        self.W2 = xavier_uniform_init((d_ff, d_model))
        self.b2 = xp.zeros((d_model,), dtype=xp.float32)

    def forward(self, x: xp.ndarray):
        """
        Executa o passo forward da rede.

        [cite_start]A equação é: FFN(x) = max(0, xW1 + b1)W2 + b2[cite: 148].

        Args:
            x (xp.ndarray): O tensor de entrada com shape (B, T, d_model).

        Returns:
            tuple[xp.ndarray, tuple]: Uma tupla contendo:
                - O tensor de saída com shape (B, T, d_model).
                - Um 'cache' com valores intermediários para o backward pass.
        """
        B, T, D = x.shape
        flat_x = x.reshape(-1, D)

        # Camada 1 + ReLU
        h_linear = flat_x @ self.W1 + self.b1
        h_relu = xp.maximum(h_linear, 0)

        # Camada 2
        out_flat = h_relu @ self.W2 + self.b2

        # Armazena valores intermediários essenciais para a retropropagação
        cache = (flat_x, h_linear, h_relu)
        return out_flat.reshape(B, T, D), cache

    def backward(self, d_out, cache):
        """
        Executa o passo backward (retropropagação) para a FFN.

        Args:
            d_out (xp.ndarray): O gradiente da perda em relação à saída desta camada.
            cache (tuple): O cache salvo durante o passo forward.

        Returns:
            tuple[xp.ndarray, dict]: Uma tupla contendo:
                - O gradiente da perda em relação à entrada da camada (d_x).
                - Um dicionário ('grads') com os gradientes de todos os pesos e biases.
        """
        flat_x, h_linear, h_relu = cache
        B, T, D = d_out.shape
        d_out_flat = d_out.reshape(-1, D)

        # 1. Gradientes da segunda camada (d_out_flat = d_h_relu @ W2 + db2)
        grad_W2 = h_relu.T @ d_out_flat
        grad_b2 = xp.sum(d_out_flat, axis=0)
        d_h_relu = d_out_flat @ self.W2.T

        # 2. Retropropagar através da ativação ReLU
        # O gradiente só passa onde a entrada original para ReLU era > 0
        d_h_linear = d_h_relu * (h_linear > 0)

        # 3. Gradientes da primeira camada (d_h_linear = d_flat_x @ W1 + db1)
        grad_W1 = flat_x.T @ d_h_linear
        grad_b1 = xp.sum(d_h_linear, axis=0)
        d_x_flat = d_h_linear @ self.W1.T

        # 4. Remodelar o gradiente de entrada para o formato original
        d_x = d_x_flat.reshape(B, T, D)

        # 5. Agrupar os gradientes dos parâmetros em um dicionário
        grads = {
            'ffn_W1': grad_W1, 'ffn_b1': grad_b1,
            'ffn_W2': grad_W2, 'ffn_b2': grad_b2
        }

        return d_x, grads
    
    def get_parameters(self) -> dict[str, xp.ndarray]:
        """Retorna um dicionário com todos os pesos e biases aprendíveis."""
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2
        }