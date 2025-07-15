import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from datasets       import load_dataset
from transformers   import AutoTokenizer
from transformer.pipeline.transformer_manual import Transformer
from transformer.pipeline.train_loop       import train_epoch
from transformer.pipeline.common           import is_gpu, xp

class NumpyDataLoader:
    def __init__(self, dataset, tokenizer, batch_size, max_len):
        self.ds         = dataset
        self.tokenizer  = tokenizer
        self.batch_size = batch_size
        self.max_len    = max_len

    def __iter__(self):
        for i in range(0, len(self.ds), self.batch_size):
            batch = self.ds[i : i + self.batch_size]["translation"]
            src = [t["en"] for t in batch]
            tgt = [t["pt"] for t in batch]
            src_ids = self.tokenizer(
                src, return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
            )["input_ids"]
            tgt_ids = self.tokenizer(
                tgt, return_tensors="np",
                padding="max_length",
                truncation=True,
                max_length=self.max_len,
            )["input_ids"]
            yield xp.asarray(src_ids), xp.asarray(tgt_ids)


def run_train(
    max_examples: int,
    epochs:       int,
    d_model:      int,
    n_layers:     int,
    n_heads:      int,
    d_ff:         int,
    max_len:      int,
    batch_size:   int,
    warmup:       int,
    checkpoint_dir:str
):
    os.makedirs(checkpoint_dir, exist_ok=True)

    # 1) tokenizer + dados
    tokenizer = AutoTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-tc-big-en-pt",
        trust_remote_code=True
    )
    vocab_size = tokenizer.vocab_size

    raw = load_dataset(
        "tatoeba", lang1="en", lang2="pt", trust_remote_code=True
    )["train"].shuffle(seed=42).select(range(max_examples))

    data_loader = NumpyDataLoader(raw, tokenizer, batch_size, max_len)

    # 2) modelo
    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len
    )
    print("GPU disponível?", is_gpu())

    # 3) training loop
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(
            model=model,
            data_loader=data_loader,
            pad_idx=tokenizer.pad_token_id,
            d_model=d_model,
            warmup=warmup
        )
        dt = time.time() - t0

        # print geral por época
        print(f"[EPOCH {epoch:02d}/{epochs:02d}] loss={loss:.4f}  time={dt:.1f}s  lr={optimizer.lr:.6f}")


        # 4) checkpoint
        params = model.get_parameters_dict()
        xp.savez(os.path.join(checkpoint_dir, f"epoch{epoch}.npz"), **params)


if __name__ == "__main__":
    # Parâmetros padrão (full train)
    run_train(
        max_examples   = 5000,
        epochs         = 10,
        d_model        = 512,
        n_layers       = 6,
        n_heads        = 8,
        d_ff           = 2048,
        max_len        = 64,
        batch_size     = 8,
        warmup         = 4000,
        checkpoint_dir = "checkpoints",
    )
