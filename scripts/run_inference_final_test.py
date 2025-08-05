import os
import sys
import numpy as onp
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformers import AutoTokenizer
from transformer.pipeline.common import xp
from transformer.pipeline.transformer_manual import Transformer

# --- 1. CONFIGURAÇÃO (a mesma do treino 'final_test') ---

# Parâmetros exatos do modelo BASE do artigo
D_MODEL = 512
N_LAYERS = 6
N_HEADS = 8
D_FF = 2048
MAX_LEN = 64
# Aponta para a pasta e o checkpoint do seu último treino
CHECKPOINT = "checkpoints_final_test/epoch10.npz" 

# --- 2. CARREGAR TOKENIZER E MODELO ---

print(f"Carregando tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(
    "Helsinki-NLP/opus-mt-tc-big-en-pt",
    trust_remote_code=True
)
vocab_size = tokenizer.vocab_size

print(f"Instanciando modelo com d_model={D_MODEL}...")
model = Transformer(
    vocab_size=vocab_size,
    d_model=D_MODEL,
    n_layers=N_LAYERS,
    n_heads=N_HEADS,
    d_ff=D_FF,
    max_len=MAX_LEN
)

# --- 3. CARREGAR CHECKPOINT ---

print(f"-> Carregando checkpoint: {CHECKPOINT}")
# Garante que o checkpoint exista antes de tentar carregar
if not os.path.exists(CHECKPOINT):
    print(f"ERRO: Checkpoint '{CHECKPOINT}' não encontrado. Verifique o caminho e o número da época.")
    sys.exit(1)

ckpt = onp.load(CHECKPOINT)
# ATENÇÃO: Se você implementou o Weight Sharing, 'Wout' não é mais um parâmetro separado.
top_level_params = ["Wemb_src", "Wemb_tgt"] 

# Lógica robusta para carregar todos os parâmetros
for name in model.get_parameters_dict().keys():
    if name in ckpt:
        if name in top_level_params:
            setattr(model, name, xp.asarray(ckpt[name]))
        else:
            parts = name.split('_')
            container_path_parts = parts[:3]
            param_name = '_'.join(parts[3:])
            target_obj = model
            for part in container_path_parts:
                if part.isdigit():
                    target_obj = target_obj[int(part)]
                else:
                    attr_map = {'sa': 'self_attn', 'ca': 'cross_attn', 'ffn': 'ffn'}
                    if 'norm' in part:
                        target_obj = getattr(target_obj, part)
                    else:
                        target_obj = getattr(target_obj, attr_map.get(part, part))
            setattr(target_obj, param_name, xp.asarray(ckpt[name]))
    else:
        print(f"!! Atenção: Parâmetro '{name}' não encontrado no checkpoint.")

print("-> Modelo carregado com sucesso.\n")

# --- 4. FUNÇÃO DE GERAÇÃO GREEDY ---

def greedy_generate(text: str) -> str:
    enc = tokenizer(
        [text], return_tensors="np", padding="max_length",
        truncation=True, max_length=MAX_LEN
    )
    src_ids = xp.asarray(enc["input_ids"])
    pad_id = tokenizer.pad_token_id

    src_mask = (src_ids == pad_id)[:, None, None, :]
    
    # O tokenizer Helsinki-NLP usa o pad_id como token inicial (SOS)
    sos_id = tokenizer.pad_token_id 
    eos_id = tokenizer.eos_token_id
    
    tgt = xp.array([[sos_id]], dtype=xp.int32)

    for _ in range(MAX_LEN - 1):
        tgt_len = tgt.shape[1]
        tgt_causal_mask = xp.triu(xp.ones((1, 1, tgt_len, tgt_len), dtype=xp.bool_), k=1)
        
        # Chama o forward com training=False para desativar o dropout
        logits = model.forward(src_ids, tgt, src_mask, tgt_causal_mask, training=False)
        
        next_id = int(xp.argmax(logits[0, -1]))
        tgt = xp.concatenate([tgt, xp.array([[next_id]], dtype=xp.int32)], axis=1)
        
        if next_id == eos_id:
            break

    out_ids = [int(i) for i in tgt[0, 1:].tolist()] # Pula o SOS inicial
    return tokenizer.decode(out_ids, skip_special_tokens=True)

# --- 5. EXEMPLOS PARA TESTE ---

test_prompts = [
    "Hello, how are you?",
    "This is a test of the Transformer implementation.",
    "Machine translation is fun!",
    "The book is on the table.",
    "What is your name?",
    "I would like to order a coffee."
]

for prompt in test_prompts:
    print(f"> {prompt}")
    translation = greedy_generate(prompt)
    print(f"< {translation}\n")