from .common import xp
from .loss import label_smoothing_loss, label_smoothing_grad
from .optimizer_scheduler import Adam, noam_schedule


def make_causal_mask(tgt_input: xp.ndarray) -> xp.ndarray:
    """
    Gera máscara causal para decoder:
    máscara triangular inferior, 0 onde permitido e -1e9 onde bloqueado.
    """
    B, T = tgt_input.shape
    # máscara triangular inferior (T, T)
    tri = xp.tril(xp.ones((T, T), dtype=xp.float32))
    mask = xp.where(tri == 1, 0.0, -1e9)
    return mask[None, None, :, :]  # shape (1, 1, T, T)


def train_epoch(model, data_loader, pad_idx: int, d_model: int, warmup: int) -> float:
    """
    Executa um passo de treinamento (uma época) no modelo Transformer.

    Args:
        model: instância do Transformer com methods forward() e get_parameters_dict().
        data_loader: iterador que retorna tuplas (src_ids, tgt_ids).
        pad_idx: índice do token de padding.
        d_model: dimensão do embedding (usado no scheduler).
        warmup: número de passos de warmup para o scheduler Noam.

    Retorna:
        loss médio sobre todos os batches.
    """
    # 1) Instancia o otimizador Adam + Noam
    params = list(model.get_parameters_dict().values())
    optimizer = Adam(params, lr=0.0)
    step_num = 0

    all_losses = []
    for batch_i, (src_ids, tgt_ids) in enumerate(data_loader):
        step_num += 1
        # atualiza learning rate via Noam schedule
        lr_t = noam_schedule(d_model, warmup, step_num)
        optimizer.lr = lr_t

        # --- Prints de debug ---
        print(f"[DEBUG][Train] Batch {batch_i} — src_ids.shape={src_ids.shape}, tgt_ids.shape={tgt_ids.shape}")
        print(f"[DEBUG][Train] src_ids[0,:10]={src_ids[0,:10].tolist()}")
        print(f"[DEBUG][Train] tgt_ids[0,:10]={tgt_ids[0,:10].tolist()}")
        
        # prepara input/output do decoder (teacher forcing)
        tgt_input = tgt_ids[:, :-1]
        tgt_output = tgt_ids[:, 1:]
        print(f"[DEBUG][Train] tgt_input.shape={tgt_input.shape}, tgt_output.shape={tgt_output.shape}")
        print(f"[DEBUG][Train] STEP={step_num}, lr={lr_t:.6e}")

        # máscara de padding no encoder
        # corrigido para shape (B,1,1,S) a partir de src_ids.shape (B,S)
        src_mask = (src_ids != pad_idx)[:, None, None, :]  # (B,1,1,S)
        non_pad = int((src_mask == 1).sum())
        print(f"[DEBUG][Train] pad_idx={pad_idx}, encoder non-pad tokens={non_pad}")

        # máscara causal para o decoder
        tgt_mask = make_causal_mask(tgt_input)
        nz = int((tgt_mask != 0).sum())
        print(f"[DEBUG][Train] tgt_mask.shape={tgt_mask.shape}, non-zero entries={nz}")

        # forward pass
        logits = model.forward(src_ids, tgt_input, src_mask, tgt_mask)
        print(f"[DEBUG][Train] logits.shape={logits.shape}  # (B, T, Vocab)")

        # cálculo de loss e backward
        loss = label_smoothing_loss(logits, tgt_output, pad_idx, epsilon=0.1)
        print(f"[DEBUG][Train] loss={loss:.4f}")

        grads = label_smoothing_grad(logits, tgt_output, pad_idx, epsilon=0.1)
        _ = optimizer.step(grads)

        all_losses.append(loss)

    # retorna perda média
    return sum(all_losses) / len(all_losses)
