from .loss                import label_smoothing_loss
from .optimizer_scheduler import Adam, noam_schedule
from .common              import xp

def create_causal_mask(size):
    """Cria uma máscara triangular para impedir atenção a tokens futuros."""
    mask = xp.triu(xp.ones((1, 1, size, size), dtype=xp.bool_), k=1)
    return mask # Shape (1, 1, size, size)

def create_padding_mask(input_ids, pad_idx):
    """Cria uma máscara para ignorar posições de padding."""
    return (input_ids == pad_idx)[:, None, None, :] # Shape (B, 1, 1, S)

def train_epoch(model, data_loader, pad_idx, d_model, warmup):
    """
    Executa uma época de treinamento do Transformer puro NumPy/CuPy.
    """
    params = list(model.get_parameters_dict().values())
    optim = Adam(list(model.get_parameters_dict().values()))

    total_loss   = 0.0
    step         = 0
    batch_count  = 0

    for src, tgt in data_loader:
        step += 1
        batch_count += 1

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
            
        # --- CORREÇÃO: CRIAÇÃO DAS MÁSCARAS ---
        src_pad_mask = create_padding_mask(src, pad_idx)
        tgt_pad_mask = create_padding_mask(tgt_input, pad_idx)
        causal_mask = create_causal_mask(tgt_input.shape[1])
                    
        # Combina máscara de padding do target com a máscara causal
        tgt_mask = tgt_pad_mask | causal_mask

        # --- CORREÇÃO: PASSAR MÁSCARAS PARA O FORWARD ---
        logits = model.forward(src, tgt_input, src_pad_mask, tgt_mask, training=True)
            
        loss = label_smoothing_loss(logits, tgt_output, pad_idx)
        total_loss += float(loss)

        grads = model.backward(logits, tgt_output, pad_idx)
        optim.lr = noam_schedule(d_model, warmup, step)
        optim.step(grads)

        # libera memória cupy (se for o caso)
        try:
            xp.get_default_memory_pool().free_all_blocks()
        except Exception:
            pass

    # retorna perda média por batch
    return total_loss / batch_count
