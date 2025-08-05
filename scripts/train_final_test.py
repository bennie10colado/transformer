import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from scripts.train import run_train

if __name__ == "__main__":
    run_train(
        max_examples   = 4096,  
        epochs         = 10,     
        batch_size     = 16,    
        
        # --- Parâmetros exatos do modelo BASE do artigo ---
        d_model        = 512,
        n_layers       = 6,
        n_heads        = 8,
        d_ff           = 2048,
        # -------------------------------------------------
        
        max_len        = 64,
        warmup         = 4000,   
        checkpoint_dir = "checkpoints_final_test",
    )