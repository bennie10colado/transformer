from .common import xp
import math

class PositionWiseFeedForwardNetwork:
    """
    Implementa a rede Feed-Forward (FFN) ponto-a-ponto do Transformer.
    Consiste em duas camadas lineares com uma ativação ReLU no meio.
    """
    def __init__(self, d_model: int, d_ff: int):
        # Inicialização de pesos (Xavier/Glorot)
        limit1 = math.sqrt(6 / (d_model + d_ff))
        limit2 = math.sqrt(6 / (d_ff + d_model))
        
        # Parâmetros aprendíveis
        self.W1 = xp.random.uniform(-limit1, limit1, (d_model, d_ff), dtype=xp.float32)
        self.b1 = xp.zeros((d_ff,), dtype=xp.float32)
        self.W2 = xp.random.uniform(-limit2, limit2, (d_ff, d_model), dtype=xp.float32)
        self.b2 = xp.zeros((d_model,), dtype=xp.float32)
        
        # Cache para a retropropagação
        self._cache = {}

    def get_parameters_dict(self) -> dict:
        """Coleta os parâmetros aprendíveis desta camada."""
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    def set_parameters_dict(self, params: dict):
        """Carrega os parâmetros aprendíveis para esta camada."""
        self.W1 = xp.asarray(params["W1"])
        self.b1 = xp.asarray(params["b1"])
        self.W2 = xp.asarray(params["W2"])
        self.b2 = xp.asarray(params["b2"])

    def forward(self, x: xp.ndarray) -> xp.ndarray:
        # Guarda a entrada para usar no backward
        self._cache['x'] = x
        
        # Primeira camada linear
        z1 = x @ self.W1 + self.b1
        
        # Ativação ReLU
        a1 = xp.maximum(0, z1)
        self._cache['a1'] = a1 # Guarda a ativação para o backward da ReLU
        
        # Segunda camada linear
        return a1 @ self.W2 + self.b2

    def backward(self, d_output: xp.ndarray):
        """
        Retropropaga o gradiente através da FFN.
        d_output: Gradiente vindo da camada seguinte.
        """
        # Desempacota valores do cache
        x = self._cache['x']
        a1 = self._cache['a1']
        
        # --- Backpropagation (ordem inversa do forward) ---

        # 1. Gradientes para W2 e b2 (da segunda camada linear)
        grad_W2 = a1.transpose(0, 2, 1).reshape(-1, a1.shape[-1]).T @ d_output.reshape(-1, d_output.shape[-1])
        grad_b2 = d_output.sum(axis=(0, 1))
        
        # 2. Retropropaga para a ativação a1
        d_a1 = d_output @ self.W2.T
        
        # 3. Retropropaga pela ativação ReLU
        # O gradiente só passa onde a ativação era > 0
        d_z1 = d_a1 * (a1 > 0)
        
        # 4. Gradientes para W1 e b1 (da primeira camada linear)
        grad_W1 = x.transpose(0, 2, 1).reshape(-1, x.shape[-1]).T @ d_z1.reshape(-1, d_z1.shape[-1])
        grad_b1 = d_z1.sum(axis=(0, 1))
        
        # 5. Gradiente final em relação à entrada x (para a camada anterior)
        d_x = d_z1 @ self.W1.T
        
        # Junta os gradientes dos pesos em um dicionário
        grads = {"W1": grad_W1, "b1": grad_b1, "W2": grad_W2, "b2": grad_b2}
        
        return d_x, grads