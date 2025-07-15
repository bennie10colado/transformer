import torch
import torch.nn.functional as F

def train_epoch(model: torch.nn.Module,
                data_loader: torch.utils.data.DataLoader,
                pad_idx: int,
                optimizer: torch.optim.Optimizer,
                scheduler: torch.optim.lr_scheduler._LRScheduler,
                device: torch.device) -> float:
    """
    Treina uma época usando PyTorch autograd.
    """
    model.train()
    total_loss = 0.0

    for batch_i, batch in enumerate(data_loader):
        src_ids = batch['src_ids'].to(device)   # (B, S)
        tgt_ids = batch['tgt_ids'].to(device)   # (B, T)

        # prepara entrada/saída do decoder (teacher forcing)
        tgt_input  = tgt_ids[:, :-1]            # (B, T-1)
        tgt_output = tgt_ids[:,  1:]            # (B, T-1)

        # máscara de padding para o encoder
        src_mask = (src_ids != pad_idx).unsqueeze(1).unsqueeze(2)  # (B,1,1,S)
        # máscara causal para o decoder
        seq_len = tgt_input.size(1)
        causal = torch.tril(torch.ones((seq_len, seq_len), device=device)).bool()
        tgt_mask = causal.unsqueeze(0).unsqueeze(1)                # (1,1,T,T)

        # forward → (B, T-1, V)
        logits = model(src_ids, tgt_input, src_mask, tgt_mask)
        B, Tm1, V = logits.shape

        # cross‑entropy ignorando pad tokens
        loss = F.cross_entropy(
            logits.reshape(-1, V),           # (B*(T-1), V)
            tgt_output.reshape(-1),          # (B*(T-1))
            ignore_index=pad_idx
        )

        # prints de debug no primeiro batch
        if batch_i == 0:
            print(f"[DEBUG][Train] src_ids.shape={src_ids.shape}, tgt_input.shape={tgt_input.shape}")
            print(f"[DEBUG][Train] first src_ids row: {src_ids[0,:10].tolist()}")
            print(f"[DEBUG][Train] first tgt_ids row: {tgt_ids[0,:10].tolist()}")
            print(f"[DEBUG][Train] logits.shape={logits.shape}, loss={loss.item():.4f}")
            print(f"[DEBUG][Train] lr={scheduler.get_last_lr()[0]:.2e}")

        # backward + otimização
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()

    return total_loss / len(data_loader)
