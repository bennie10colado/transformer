from .common import xp

def log_softmax(x):
    """
    log-softmax estável em NumPy
    """
    x_shift = x - xp.max(x, axis=-1, keepdims=True)
    log_sum = xp.log(xp.sum(xp.exp(x_shift), axis=-1, keepdims=True))
    return x_shift - log_sum

def softmax(x: xp.ndarray, grad: xp.ndarray | None = None) -> xp.ndarray:
    """
    Calcula o softmax (para o forward pass) ou o gradiente do softmax 
    (para o backward pass).
    """
    if grad is None:
        # --- FORWARD PASS ---
        # Subtrai o máximo para estabilidade numérica
        e_x = xp.exp(x - xp.max(x, axis=-1, keepdims=True))
        return e_x / xp.sum(e_x, axis=-1, keepdims=True)
    else:
        # --- BACKWARD PASS ---
        # grad aqui é dL/dy, onde y é a saída do softmax
        # x aqui é y, a saída do softmax do forward pass
        y = x
        # Calcula o Jacobiano (dy/dx) multiplicado pelo gradiente de entrada (dL/dy)
        # de forma eficiente.
        # dL/dx_i = y_i * (dL/dy_i - sum(dL/dy_j * y_j))
        sum_grad_y = xp.sum(grad * y, axis=-1, keepdims=True)
        return y * (grad - sum_grad_y)

def projection(x, W, b):
    return x @ W + b
