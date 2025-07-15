from .common import xp
from .optimizer_scheduler import Adam

def train_epoch(model, data_loader, pad_idx, **kwargs): # kwargs para compatibilidade
    total_loss = 0.
    
    param_dict = model.get_parameters_dict()
    params = list(param_dict.values())
    param_keys = list(param_dict.keys())

    # AJUSTE 1: Taxa de aprendizado (learning rate) menor e mais segura.
    optim = Adam(params, lr=3e-5) # Valor muito mais estável que 1e-4 para começar
    
    for i, (src_batch, tgt_batch) in enumerate(data_loader):
        tgt_input = tgt_batch[:, :-1]
        tgt_output = tgt_batch[:, 1:]

        # --- Forward Pass ---
        logits = model.forward(src_batch, tgt_input)
        
        # --- Cálculo do Loss (Mais Estável) ---
        B, T, V = logits.shape
        
        # AJUSTE 2: Usando log-softmax para estabilidade numérica
        # Log-Softmax
        log_probs = logits - xp.log(xp.sum(xp.exp(logits), axis=-1, keepdims=True))
        
        # Negative Log-Likelihood Loss (NLLLoss)
        # Seleciona os log_probs correspondentes às palavras corretas
        nll_loss = -xp.take_along_axis(log_probs, tgt_output.reshape(B, T, 1), axis=2).squeeze()
        
        # Ignora o loss de tokens de padding
        non_pad_mask = (tgt_output != pad_idx)
        nll_loss *= non_pad_mask
        
        # Calcula a média do loss apenas para os tokens válidos
        loss = nll_loss.sum() / non_pad_mask.sum()
        total_loss += loss.item()

        # --- Backward Pass ---
        grads_dict = model.backward(log_probs, tgt_output, pad_idx)

        # --- Conversão e Passo do Otimizador ---
        grads_list = [grads_dict.get(key) for key in param_keys]
        optim.step(grads_list)

    return total_loss / (i + 1)