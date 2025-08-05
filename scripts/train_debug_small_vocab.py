import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from transformer.pipeline.transformer_manual import Transformer
from transformer.pipeline.train_loop import train_epoch
from transformer.pipeline.common import is_gpu, xp

# --- 1. DATASET MIN E CRIAÇÃO DE VOCABULÁRIO ---

TINY_DATASET = [
    {"en": "hello world", "pt": "ola mundo"},
    {"en": "how are you", "pt": "como voce esta"},
    {"en": "this is a test", "pt": "isto e um teste"},
    {"en": "thank you", "pt": "obrigado"},
    {"en": "a small cat", "pt": "um gato pequeno"},
    {"en": "a big dog", "pt": "um cao grande"},
    {"en": "good morning", "pt": "bom dia"},
    {"en": "good night", "pt": "boa noite"},
    {"en": "i love machine translation", "pt": "eu amo traducao de maquina"},
    {"en": "the book is on the table", "pt": "o livro esta na mesa"},
]

# Construindo o vocabulário
def build_vocab(dataset):
    # Tokens especiais são essenciais
    vocab = {'<pad>': 0, '<sos>': 1, '<eos>': 2, '<unk>': 3}
    idx = 4
    for pair in dataset:
        for sentence in [pair["en"], pair["pt"]]:
            for word in sentence.split():
                if word not in vocab:
                    vocab[word] = idx
                    idx += 1
    return vocab

# Nosso vocabulário customizado
vocab = build_vocab(TINY_DATASET)
vocab_size = len(vocab)
pad_idx = vocab['<pad>']
print(f"Vocabulário customizado criado com {vocab_size} tokens.")

# --- 2. DATALOADER CUSTOMIZADO ---

class CustomNumpyDataLoader:
    def __init__(self, dataset, vocab, batch_size, max_len):
        self.dataset = dataset
        self.vocab = vocab
        self.batch_size = batch_size
        self.max_len = max_len
        self.pad_idx = vocab['<pad>']
        self.sos_idx = vocab['<sos>']
        self.eos_idx = vocab['<eos>']

    def tokenize(self, sentence):
        # Tokeniza uma sentença e adiciona SOS/EOS
        tokens = [self.sos_idx]
        for word in sentence.split():
            tokens.append(self.vocab.get(word, self.vocab['<unk>']))
        tokens.append(self.eos_idx)
        return tokens

    def pad(self, tokens):
        # Aplica padding até o max_len
        padded = tokens + [self.pad_idx] * (self.max_len - len(tokens))
        return padded[:self.max_len]

    def __iter__(self):
        for i in range(0, len(self.dataset), self.batch_size):
            batch = self.dataset[i : i + self.batch_size]
            src_batch = [self.pad(self.tokenize(p["en"])) for p in batch]
            tgt_batch = [self.pad(self.tokenize(p["pt"])) for p in batch]
            yield xp.asarray(src_batch, dtype=xp.int32), xp.asarray(tgt_batch, dtype=xp.int32)

# --- 3. EXECUÇÃO DO TREINO ---

def run_custom_train():
    # Parâmetros de debug
    d_model = 64
    n_layers = 2
    n_heads = 2
    d_ff = 128
    max_len = 20 # Podemos usar um max_len menor
    batch_size = 2
    epochs = 50 # Treine por mais épocas para decorar o dataset
    warmup = 10
    checkpoint_dir = "checkpoints_small_vocab2"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Salva o vocabulário para ser usado na inferência depois
    np.save(os.path.join(checkpoint_dir, "vocab.npy"), vocab)

    data_loader = CustomNumpyDataLoader(TINY_DATASET, vocab, batch_size, max_len)

    model = Transformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        d_ff=d_ff,
        max_len=max_len
    )
    print("GPU disponível?", is_gpu())

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        loss = train_epoch(
            model=model,
            data_loader=data_loader,
            pad_idx=pad_idx,
            d_model=d_model,
            warmup=warmup
        )
        dt = time.time() - t0
        print(f"[Ep {epoch}/{epochs}] loss={loss:.4f}  time={dt:.1f}s")

        if epoch % 10 == 0:
            params = model.get_parameters_dict()
            xp.savez(os.path.join(checkpoint_dir, f"epoch{epoch}.npz"), **params)
    print("Treino com vocabulário pequeno concluído!")


if __name__ == "__main__":
    run_custom_train()