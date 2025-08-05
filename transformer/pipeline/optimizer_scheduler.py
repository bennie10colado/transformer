from .common import xp

class Adam:
    def __init__(self, params):
        # Parâmetros EXATOS do artigo 
        self.params = params
        self.lr = 1.0 # O LR é controlado externamente pela schedule
        self.beta1 = 0.9
        self.beta2 = 0.98
        self.eps = 1e-9
        self.m = [xp.zeros_like(p) for p in self.params]
        self.v = [xp.zeros_like(p) for p in self.params]
        self.step_num = 0

    def step(self, grads):
        self.step_num += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (g ** 2)
            
            m_hat = self.m[i] / (1 - self.beta1 ** self.step_num)
            v_hat = self.v[i] / (1 - self.beta2 ** self.step_num)
            
            update_value = self.lr * m_hat / (xp.sqrt(v_hat) + self.eps)
            p -= update_value

def noam_schedule(d_model: int, warmup_steps: int, step_num: int):
    """
    A implementação EXATA da schedule de learning rate do artigo.
    """
    if step_num == 0:
        step_num = 1
        
    arg1 = step_num ** -0.5
    arg2 = step_num * (warmup_steps ** -1.5)
    
    return (d_model ** -0.5) * min(arg1, arg2)