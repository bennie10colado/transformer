import sys, os, time, csv, psutil
from tqdm import tqdm

import torch
from transformers import AutoTokenizer
from datasets import load_dataset
import evaluate

# adiciona a raiz do projeto no PYTHONPATH
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, ROOT)

from transformer.pipeline.common             import xp, is_gpu
from transformer.pipeline.transformer_manual import Transformer
from transformer.pipeline.inference.translator import Translator

def compute_metrics(dataset, translator, tokenizer, num_samples=1000, batch_size=4):
    references, hypotheses = [], []

    for i in tqdm(range(0, min(num_samples, len(dataset)), batch_size),
                  desc="Calculando métricas"):
        # pega o slice como Dataset
        batch = dataset[i : i + batch_size]
        # agora itera sobre o campo "translation", que é uma lista de dicts
        translations = batch["translation"]
        src_texts = [t["en"] for t in translations]
        tgt_texts = [t["pt"] for t in translations]

        # traduz cada sentença via Translator
        for src in src_texts:
            hyp = translator.translate(
                src,
                method="beam",
                beam_size=4,
                max_len=tokenizer.model_max_length
            )
            hypotheses.append(hyp)

        references.extend([[t] for t in tgt_texts])

    bleu = evaluate.load("bleu")
    chrf = evaluate.load("chrf")
    bleu_score = bleu.compute(predictions=hypotheses, references=references)["bleu"] * 100
    chrf_score = chrf.compute(predictions=hypotheses, references=references)["score"]
    return bleu_score, chrf_score

def main():
    # 0) dispositivo só para torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando GPU? {torch.cuda.is_available()}")
    print(f"Dispositivo: {device}")

    # 1) carregando dataset
    ds = load_dataset("tatoeba", lang1="en", lang2="pt", trust_remote_code=True)["train"]
    ds = ds.select(range(10))
    print(f"Total de sentenças: {len(ds)}")
    print(f"Exemplo de sentença: {ds[0]}")
    print(f"Total de palavras: {sum(len(ex['translation']['pt'].split()) for ex in ds)}")
    
    # 2) tokenizer
    model_name = "Helsinki-NLP/opus-mt-tc-big-en-pt"
    tokenizer  = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    vocab_size = tokenizer.vocab_size

    # 3) instanciando Transformer puro NumPy/CuPy
    model = Transformer(
        vocab_size=vocab_size,
        d_model=512,
        n_layers=6,
        n_heads=8,
        d_ff=2048,
        max_len=tokenizer.model_max_length
    )
    # só chama .to() se for nn.Module
    if hasattr(model, "to"):
        model = model.to(device)

    # 4) prepara o Translator
    stoi = tokenizer.get_vocab()
    itos = {v:k for k,v in stoi.items()}
    translator = Translator(
        model=model,
        stoi=stoi,
        itos=itos,
        pad_idx=tokenizer.pad_token_id,
        sos_idx=tokenizer.bos_token_id or tokenizer.cls_token_id,
        eos_idx=tokenizer.eos_token_id or tokenizer.sep_token_id
    )

    # 5) avalia
    mem0 = psutil.Process().memory_info().rss
    t0   = time.time()
    bleu_score, chrf_score = compute_metrics(ds, translator, tokenizer)
    dt   = time.time() - t0
    mem1 = psutil.Process().memory_info().rss

    n_sent = len(ds)
    n_words= sum(len(ex["translation"]["pt"].split()) for ex in ds)
    ram_mb = (mem1 - mem0) / (1024**2)

    print("\n" + "-"*60)
    print(f"BLEU:         {bleu_score:.2f}")
    print(f"chr-F:        {chrf_score:.2f}")
    print(f"Tempo total:  {dt:.2f}s (≈{dt/n_sent:.4f}s/sent)")
    print(f"RAM Δ:        {ram_mb:.2f} MB")
    print(f"Palavras/s:   {n_words/dt:.2f}")
    print("-"*60 + "\n")

    # 6) salva CSV
    out_file = "resultados_traducao.csv"
    header = ["Modelo","Tempo(s)","Tempo/sent","Palavras/s","RAM(MB)","Sentenças","Palavras","BLEU","chr-F"]
    if not os.path.exists(out_file):
        with open(out_file, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(header)
    with open(out_file, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow([
            model_name,
            f"{dt:.2f}",
            f"{dt/n_sent:.4f}",
            f"{n_words/dt:.2f}",
            f"{ram_mb:.2f}",
            n_sent,
            n_words,
            f"{bleu_score:.2f}",
            f"{chrf_score:.2f}",
        ])

if __name__ == "__main__":
    main()
