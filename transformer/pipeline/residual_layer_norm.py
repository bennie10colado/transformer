from .common import xp

class ResidualLayerNorm:
    # A equação de Layer Normalization vem da seção 3.1 do artigo
    # "Attention Is All You Need" (Vaswani et al., 2017),
    # que toma como referência outro artigo:
    # "Layer Normalization" (Ba, J. L., Kiros, J. R., & Hinton, G. E., 2016),
    # a fórmula é LayerNorm(x+Sublayer(x)).
    #
    # Para o dropout foi usado como referência a seção 5.4 do artigo
    # "Attention Is All You Need" (Vaswani et al., 2017),
    # que menciona o uso de dropout antes da normalização da camada.
    def __init__(self, d_model: int, dropout: float = 0.1, eps: float = 1e-6):
        """
        d_model: dimensão dos embeddings
        dropout: probabilidade de zerar ativações da subcamada
        eps: termo de estabilidade numérica
        """
        self.dropout_rate = dropout
        self.eps     = eps
        # parâmetros aprendíveis γ e β
        # para a normalização: gamma (escala) e beta (deslocamento)
        self.gamma = xp.ones((d_model,), dtype=xp.float32)
        self.beta = xp.zeros((d_model,), dtype=xp.float32)

    '''def forward_old(self, x: xp.ndarray, sublayer_out: xp.ndarray) -> xp.ndarray:
        """
        Aplica conexão residual + dropout + layer norm.

        Args:
            x:           tensor de entrada (batch, seq_len, d_model)
            sublayer_out: saída da subcamada (mesma forma que x)
        Returns:
            tensor normalizado, mesma forma de x
        """
        # aplica dropout à saída da subcamada
        if self.dropout > 0.0:
            mask = (xp.random.rand(*sublayer_out.shape) >= self.dropout).astype(xp.float32)
            sub = sublayer_out * mask / (1.0 - self.dropout)
        else:
            sub = sublayer_out

        # conexão residual
        y = x + sub

        # cálculo do LayerNorm
        mean = xp.mean(y, axis=-1, keepdims=True)
        var  = xp.mean((y - mean) ** 2, axis=-1, keepdims=True)
        y_norm = (y - mean) / xp.sqrt(var + self.eps)

        # escala e shift
        return self.gamma * y_norm + self.beta'''

    def forward(self, x: xp.ndarray, sublayer_out: xp.ndarray, training: bool = True):
        """
        Aplica a sequência: Dropout -> Conexão Residual -> Layer Normalization.

        Args:
            x (xp.ndarray): A entrada original da subcamada (antes da atenção ou FFN).
            sublayer_out (xp.ndarray): A saída da subcamada.
            training (bool): Flag que indica se o modelo está em modo de treino (para ativar o dropout).

        Returns:
            tuple[xp.ndarray, dict]: A saída final da camada e um cache para o backward.
        """
        dropout_mask = None
        sub_dropped = sublayer_out

        # 1. Aplica Dropout (apenas durante o treino)
        if self.dropout_rate > 0.0 and training:
            dropout_mask = (xp.random.rand(*sublayer_out.shape) >= self.dropout_rate).astype(xp.float32)
            sub_dropped = sublayer_out * dropout_mask / (1.0 - self.dropout_rate)

        # 2. Conexão Residual
        y = x + sub_dropped

        # 3. Layer Normalization
        mean = xp.mean(y, axis=-1, keepdims=True)
        var = xp.var(y, axis=-1, keepdims=True)
        std = xp.sqrt(var + self.eps)
        y_norm = (y - mean) / std
        
        # 4. Escala e Deslocamento
        out = self.gamma * y_norm + self.beta
        
        # Salva todos os valores intermediários necessários para a retropropagação
        cache = (x, sub_dropped, dropout_mask, y, mean, std, y_norm, self.gamma)
        return out, cache

    def backward(self, d_out, cache):
        """
        Calcula os gradientes para a camada ResidualLayerNorm.

        Args:
            d_out (xp.ndarray): O gradiente vindo da camada seguinte.
            cache (dict): O cache salvo durante o passo forward.

        Returns:
            tuple[xp.ndarray, xp.ndarray, dict]: Gradientes para a entrada x, para a
            saída da subcamada e para os parâmetros gamma e beta.
        """
        x, sub_dropped, dropout_mask, y, mean, std, y_norm, gamma = cache
        
        # Gradientes dos parâmetros gamma e beta
        grad_gamma = xp.sum(d_out * y_norm, axis=(0, 1))
        grad_beta = xp.sum(d_out, axis=(0, 1))
        
        # Backprop pela normalização (a parte matematicamente mais complexa)
        d_y_norm = d_out * gamma
        
        D = x.shape[-1]
        d_var = xp.sum(d_y_norm * (y - mean) * (-0.5) * (std ** -3), axis=-1, keepdims=True)
        d_mean = xp.sum(d_y_norm * (-1/std), axis=-1, keepdims=True) + d_var * xp.mean(-2 * (y - mean), axis=-1, keepdims=True)
        
        dy = (d_y_norm / std) + (d_var * 2 * (y - mean) / D) + (d_mean / D)
        
        # Backprop pela conexão residual (o gradiente se divide igualmente)
        dx = dy
        d_sub_dropped = dy
        
        # Backprop pelo dropout
        d_sublayer_out = d_sub_dropped
        if self.dropout_rate > 0.0 and dropout_mask is not None:
            d_sublayer_out = d_sub_dropped * dropout_mask / (1.0 - self.dropout_rate)
            
        # CORREÇÃO FINAL: Retorna um dicionário com os nomes de gradientes corretos ('gamma', 'beta')
        grads = {'gamma': grad_gamma, 'beta': grad_beta}
        
        return dx, d_sublayer_out, grads