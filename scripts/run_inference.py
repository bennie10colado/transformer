import os
import sys
import numpy as onp

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets import load_dataset
from transformers import AutoTokenizer
from transformer.pipeline.common import xp
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
    CHECKPOINT = "checkpoints/epoch2.npz"
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

# 3) Carrega checkpoint
print(f"→ Usando {'DEBUG' if DEBUG else 'FULL'} checkpoint: {CHECKPOINT}")
print(f"GPU disponível? {hasattr(xp, 'get_default_memory_pool')}")

try:
    # Carrega o arquivo de pesos
    weights = onp.load(CHECKPOINT, allow_pickle=True)
    
    # Extrai o dicionário de parâmetros do arquivo .npz
    if len(weights.files) == 1 and weights.files[0] == 'arr_0':
        params_dict = weights['arr_0'].item()
    else:
        params_dict = dict(weights)
    
    # === MUDANÇA PRINCIPAL ===
    # Usa o novo método para carregar TODOS os pesos do modelo de uma só vez.
    model.set_parameters_dict(params_dict) 
    
    print("Pesos carregados com sucesso via set_parameters_dict.")

except Exception as e:
    print(f"Erro ao carregar os pesos: {e}")
    # sys.exit(1)


# 4) Função de geração *greedy*
def greedy_generate(text: str) -> str:
    enc = tokenizer(
        [text],
        return_tensors="np",
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )
    src_ids = xp.asarray(enc["input_ids"])
    pad_id  = tokenizer.pad_token_id
    is_pad  = (src_ids == pad_id)
    mask_vals = xp.where(is_pad, -1e9, 0.0).astype(xp.float32)
    src_mask = mask_vals[:, None, None, :]
    sos = (tokenizer.bos_token_id or tokenizer.cls_token_id or pad_id)
    eos = (tokenizer.eos_token_id or tokenizer.sep_token_id or sos)
    tgt = xp.array([[sos]], dtype=xp.int32)

    for _ in range(MAX_LEN - 1):
        logits = model.forward(src_ids, tgt, src_mask, None)
        next_id = int(xp.argmax(logits[0, -1]))
        tgt = xp.concatenate(
            [tgt, xp.array([[next_id]], dtype=xp.int32)], axis=1
        )
        if next_id == eos:
            break
            
    out_ids = [int(i) for i in tgt[0, 1:].tolist()]
    return tokenizer.decode(out_ids, skip_special_tokens=True)

# 5) Carrega o dataset e testa nos dados de treino
print("\n--- Testando Overfitting nos Dados de Treino ---\n")

dataset_de_treino = load_dataset(
    "tatoeba", lang1="en", lang2="pt", trust_remote_code=True
)["train"].select(range(10))

for i, exemplo in enumerate(dataset_de_treino):
    frase_en = exemplo["translation"]["en"]
    frase_pt_real = exemplo["translation"]["pt"]

    # Gera a tradução com o modelo
    frase_pt_modelo = greedy_generate(frase_en)

    print(f"--- Exemplo #{i+1} ---")
    print(f"Entrada (en):         {frase_en}")
    print(f"Tradução REAL (pt):   {frase_pt_real}")
    print(f"Tradução MODELO (pt): {frase_pt_modelo}\n")