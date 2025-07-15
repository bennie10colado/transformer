import os
import sys

# Garante que o Python encontre o pacote 'transformer'
project_root = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, project_root)

from scripts.train import run_train

if __name__ == "__main__":
    run_train(
        max_examples   = 1024,    # 1 k amostras
        epochs         =   3,     # 3 épocas
        d_model        = 128,     # dimensão 128
        n_layers       =   2,     # 2 camadas
        n_heads        =   4,     # 4 cabeças
        d_ff           = 512,     # feed‑forward 512
        max_len        =  64,     # sequência até 64 tokens
        batch_size     =  16,     # batch de 16
        warmup         =  800,    # warmup aumentado
        checkpoint_dir = "checkpoints_debug",
    )
