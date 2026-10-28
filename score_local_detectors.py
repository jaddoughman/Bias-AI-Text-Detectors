"""Score texts with the open-source detectors at pinned HuggingFace revisions.

Two preprocessing modes are supported:

  original  Replicates the wrapper used for the September 2025 runs
            (text_detector.py at the repository root): the text is tokenised with
            truncation at model.config.max_position_embeddings (514 on these
            RoBERTa checkpoints), decoded back to a string, and passed to the model
            again without truncation. Inputs that still exceed the positional limit
            raise an IndexError in the model; the original loop caught the exception
            and skipped the text. We record such texts as failed ("ERROR").
  standard  Tokenise once with truncation at 512 tokens (the architectural maximum)
            and score. Every text receives a score.

P(AI) is taken from the softmax index that the publisher designates as
machine-generated (see AI_INDEX and S5 Table).

Usage:
  python score_local_detectors.py --input texts.csv --text-col Text --id-col text_id \
      --detectors hello fakespot roberta_large radar --mode standard --output out.csv
"""
import argparse
import time

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DETECTORS = {
    # key: (checkpoint, full revision hash, AI softmax index, slow tokenizer?)
    "hello": ("Hello-SimpleAI/chatgpt-detector-roberta",
              "d2b342c61775d5dd0221808a79983ed3b86ffd86", 1, False),   # id2label {0: Human, 1: ChatGPT}
    "fakespot": ("fakespot-ai/roberta-base-ai-text-detection-v1",
                 "f9cdb14d1f8b105f597d80fa7b56f20c6ea0e9db", 1, True),  # id2label {0: Human, 1: AI}
    "roberta_large": ("openai-community/roberta-large-openai-detector",
                      "38f3e0ccf205e9c4e0b38ae9d75ec948141bf832", 0, False),  # server.py: fake, real = probs
    "radar": ("TrustSafeAI/RADAR-Vicuna-7B",
              "4ff1f23a69a36aa1df47b0933be6279f1b896c9b", 0, False),   # app.py: softmax[:, 0] = AI
    "e5": ("MayZhou/e5-small-lora-ai-generated-detector",
           "483fc4969592dc20e00e5130e7187b5dd25dbcc7", 1, False),      # model card: Label_1 = AI
}


def load(key):
    ckpt, rev, ai_idx, slow = DETECTORS[key]
    tok = AutoTokenizer.from_pretrained(ckpt, revision=rev, use_fast=not slow)
    model = AutoModelForSequenceClassification.from_pretrained(ckpt, revision=rev).eval()
    return tok, model, ai_idx


@torch.no_grad()
def score_standard(texts, tok, model, ai_idx, batch_size=16):
    out = []
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-bucketed batches
    res = {}
    for s in range(0, len(order), batch_size):
        idx = order[s:s + batch_size]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=512,
                  padding=True, return_tensors="pt")
        p = torch.softmax(model(**enc).logits.float(), -1)[:, ai_idx].tolist()
        n_tok = enc["attention_mask"].sum(1).tolist()
        for i, pi, nt in zip(idx, p, n_tok):
            res[i] = (pi, nt, "")
    for i in range(len(texts)):
        out.append(res[i])
    return out


@torch.no_grad()
def score_original(texts, tok, model, ai_idx):
    max_len = model.config.max_position_embeddings
    out = []
    for t in texts:
        ids = tok.encode(t, truncation=True, max_length=max_len)
        t2 = tok.decode(ids, skip_special_tokens=True)
        enc = tok(t2, return_tensors="pt")
        try:
            p = torch.softmax(model(**enc).logits.float(), -1)[0, ai_idx].item()
            out.append((p, enc["input_ids"].shape[1], ""))
        except Exception as e:  # positional-embedding overflow, as in the original runs
            out.append((float("nan"), enc["input_ids"].shape[1], f"ERROR: {type(e).__name__}"))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--text-col", default="Text")
    ap.add_argument("--id-col", default="text_id")
    ap.add_argument("--detectors", nargs="+", default=list(DETECTORS))
    ap.add_argument("--mode", choices=["standard", "original"], default="standard")
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    df = pd.read_csv(args.input)
    texts = df[args.text_col].astype(str).tolist()
    res = df[[args.id_col]].copy()
    for key in args.detectors:
        t0 = time.time()
        tok, model, ai_idx = load(key)
        fn = score_standard if args.mode == "standard" else score_original
        r = fn(texts, tok, model, ai_idx) if args.mode == "original" else fn(texts, tok, model, ai_idx)
        res[f"{key}_p_ai"] = [x[0] for x in r]
        res[f"{key}_n_tokens"] = [x[1] for x in r]
        res[f"{key}_pred_ai"] = [(x[0] > 0.5) if x[0] == x[0] else pd.NA for x in r]
        res[f"{key}_error"] = [x[2] for x in r]
        print(f"{key}: {len(texts)} texts in {time.time() - t0:.0f}s; "
              f"errors={sum(1 for x in r if x[2])}; flagged={sum(1 for x in r if x[0] == x[0] and x[0] > 0.5)}",
              flush=True)
        del model
        res.to_csv(args.output, index=False)
    res.to_csv(args.output, index=False)


if __name__ == "__main__":
    main()
