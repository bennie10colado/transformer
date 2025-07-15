import os
import sys
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformers import AutoTokenizer
from transformer.pipeline.common            import xp
from transformer.pipeline.transformer_manual import Transformer

# ─────── CONFIGURAÇÃO ───────────────────────────────────────────────
DEBUG = True

if DEBUG:
    D_MODEL    = 128
    N_LAYERS   = 2
    N_HEADS    = 4
    D_FF       = 512
    MAX_LEN    = 64
    CHECKPOINT = "checkpoints_debug/epoch3.npz" 
else:
    D_MODEL    = 512
    N_LAYERS   = 6
    N_HEADS    = 8
    D_FF       = 2048
    MAX_LEN    = 64
    CHECKPOINT = "checkpoints/epoch9.npz"
# ───────────────────────────────────────────────────────────────────

# 1) Carrega tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    trust_remote_code=True
)
vocab_size = tokenizer.vocab_size

# 2) Instancia o Transformer
model = Transformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
    max_len=MAX_LEN
)

# 3) Carrega checkpoint NumPy
ckpt = onp.load(CHECKPOINT)
model.Wemb_src = xp.asarray(ckpt["Wemb_src"])
model.Wemb_tgt = xp.asarray(ckpt["Wemb_tgt"])
model.Wout     = xp.asarray(ckpt["Wout"])
# Se tiver treinado mais parametros na versão FULL:
# ex: model.enc[0].self_attn.W_q = xp.asarray(ckpt["enc_0_Wq"]) etc.

print(f"→ Usando {'DEBUG' if DEBUG else 'FULL'} checkpoint: {CHECKPOINT}")
print("GPU disponível?", hasattr(xp, "get_default_memory_pool"))

# 4) Função de geração de words *greedy*
def greedy_generate(text: str) -> str:
    # tokenização
    enc = tokenizer(
        [text],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )
    src_ids = xp.asarray(enc["input_ids"])
    pad_id   = tokenizer.pad_token_id
    print(f"[DEBUG][Infer] prompt=\"{text}\"")
    print(f"[DEBUG][Infer] src_ids={src_ids.tolist()}")
    print(f"[DEBUG][Infer] pad_id={pad_id}, pad positions={(src_ids==pad_id).sum()}")

    # máscara de cross‑attention
    is_pad   = (src_ids == pad_id)
    mask_vals = xp.where(is_pad, -1e9, 0.0).astype(xp.float32)
    src_mask = mask_vals[:, None, None, :]
    print(f"[DEBUG][Infer] src_mask[0,0,0,:10]={src_mask[0,0,0,:10].tolist()}")

    # sequências alvo
    sos = tokenizer.bos_token_id or tokenizer.cls_token_id or pad_id
    eos = tokenizer.eos_token_id or tokenizer.sep_token_id or sos
    tgt = xp.array([[sos]], dtype=xp.int32)

    for t in range(MAX_LEN - 1):
        logits = model.forward(src_ids, tgt, src_mask, None)  # (1, t+1, V)
        probs  = softmax(logits[0, -1, :])                    # (V,)
        next_id = int(xp.argmax(probs))
        next_prob = float(probs[next_id])
        print(f"[DEBUG][Infer] step={t:02d} next_id={next_id} prob={next_prob:.4f}")

        tgt = xp.concatenate([tgt, xp.array([[next_id]], dtype=xp.int32)], axis=1)
        if next_id == eos:
            print(f"[DEBUG][Infer] EOS token reached at step {t}")
            break

    out_ids = tgt[0, 1:].tolist()
    print(f"[DEBUG][Infer] out_ids={out_ids}")
    return tokenizer.decode(out_ids, skip_special_tokens=True)


# 5) Exemplos
for prompt in [
    "Hello, how are you?",
    "This is a test of the Transformer implementation.",
    "Machine translation is fun!"
]:
    print(f"> {prompt}")
    print(f"< {greedy_generate(prompt)}\n")
