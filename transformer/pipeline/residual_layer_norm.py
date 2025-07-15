from .common import xp

class ResidualLayerNorm:
    """
    Implementa a conexão residual, dropout e Layer Normalization.
    A fórmula é: LayerNorm(x + Dropout(Sublayer(x)))
    """
    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-5):
        self.dropout = dropout
        self.eps = eps
        
        # Parâmetros aprendíveis: gamma (escala) e beta (deslocamento)
        self.gamma = xp.ones((d_model,), dtype=xp.float32)
        self.beta = xp.zeros((d_model,), dtype=xp.float32)
        
        # Cache para guardar valores para a retropropagação
        self._cache = {}

    def get_parameters_dict(self) -> dict:
        """Coleta os parâmetros aprendíveis desta camada."""
        return {"gamma": self.gamma, "beta": self.beta}

    def set_parameters_dict(self, params: dict):
        """Carrega os parâmetros aprendíveis para esta camada."""
        self.gamma = xp.asarray(params["gamma"])
        self.beta = xp.asarray(params["beta"])

    def forward(self, x: xp.ndarray, sublayer_out: xp.ndarray, training: bool = True) -> xp.ndarray:
        """
        Aplica a sequência: dropout -> conexão residual -> layer norm.
        """
        # 1. Aplica dropout à saída da subcamada (apenas durante o treino)
        sub_dropped = sublayer_out
        if training and self.dropout > 0.0:
            # Usamos inverted dropout
            dropout_mask = (xp.random.rand(*sublayer_out.shape) >= self.dropout)
            sub_dropped = sublayer_out * dropout_mask / (1.0 - self.dropout)
            self._cache['dropout_mask'] = dropout_mask
        
        # 2. Conexão residual
        y = x + sub_dropped
        self._cache['y'] = y # Entrada para a normalização

        # 3. Layer Normalization
        mean = y.mean(axis=-1, keepdims=True)
        var = y.var(axis=-1, keepdims=True)
        y_norm = (y - mean) / xp.sqrt(var + self.eps)
        
        # Cache para o backward
        self._cache['y_norm'] = y_norm
        self._cache['var'] = var
        
        # 4. Escala e deslocamento final com gamma e beta
        return self.gamma * y_norm + self.beta

    def backward(self, d_output: xp.ndarray):
        """
        Retropropaga o gradiente através da camada.
        d_output: Gradiente vindo da camada seguinte.
        """
        # Desempacota valores do cache
        y_norm = self._cache['y_norm']
        y = self._cache['y']
        var = self._cache['var']
        D = y.shape[-1]

        # ---- Retropropagação passo a passo (ordem inversa do forward) ----

        # 4. Gradientes de gamma e beta (parâmetros aprendíveis)
        grad_gamma = (d_output * y_norm).sum(axis=(0, 1))
        grad_beta = d_output.sum(axis=(0, 1))
        
        # Gradiente que flui para a etapa de normalização
        d_y_norm = d_output * self.gamma
        
        # 3. Retropropaga pela Layer Normalization
        inv_std = 1. / xp.sqrt(var + self.eps)
        d_var = -0.5 * (d_y_norm * (y - y.mean(axis=-1, keepdims=True)) * (inv_std**3)).sum(axis=-1, keepdims=True)
        d_mean = (-d_y_norm * inv_std).sum(axis=-1, keepdims=True)
        d_y = (d_y_norm * inv_std) + (d_var * 2 * (y - y.mean(axis=-1, keepdims=True)) / D) + (d_mean / D)
        
        # 2. Retropropaga pela conexão residual
        # O gradiente se divide igualmente para 'x' e 'sub_dropped'
        d_x = d_y
        d_sub_dropped = d_y
        
        # 1. Retropropaga pelo dropout
        d_sublayer_output = d_sub_dropped
        if 'dropout_mask' in self._cache:
            d_sublayer_output = d_sub_dropped * self._cache['dropout_mask'] / (1.0 - self.dropout)

        # Empacota os gradientes dos parâmetros
        grads = {"gamma": grad_gamma, "beta": grad_beta}
        
        # Retorna os gradientes para as entradas e para os pesos
        return d_x, d_sublayer_output, grads