from .common import xp

class Adam:
    def __init__(self, params: list, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        
        # Inicializa os momentos de primeira e segunda ordem
        self.m = [xp.zeros_like(p) for p in self.params]
        self.v = [xp.zeros_like(p) for p in self.params]

    def step(self, grads: list):
        """
        Executa um passo de otimização.
        Espera uma LISTA de gradientes, na mesma ordem dos parâmetros.
        """
        self.t += 1
        
        for i, (p, g) in enumerate(zip(self.params, grads)):
            if g is None:
                continue

            # Atualiza os momentos
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            # Correção de viés (bias correction)
            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)
            
            # Atualiza o parâmetro
            p -= self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)