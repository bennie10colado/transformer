try:
    import cupy as xp
    _GPU = True
except ImportError:
    import numpy as xp
    _GPU = False

def is_gpu():
    return _GPU

import math

def xavier_uniform_init(shape, dtype=xp.float32):
    """Implementa a inicialização de pesos Xavier/Glorot."""
    # Calcula os limites para a distribuição uniforme
    limit = math.sqrt(6 / (shape[0] + shape[1]))
    # Gera pesos em uma distribuição uniforme entre -limit e +limit
    return (xp.random.rand(*shape, dtype=dtype) * 2 - 1) * limit