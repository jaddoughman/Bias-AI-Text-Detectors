"""Re-run the excluded open-source detectors on the ICNALE human-authored corpus.

Produces the baseline misclassification rates reported in Supplementary Table S4,
under BOTH label mappings:

  * "as screened" - the HuggingFace library default (LABEL_0 -> Human, LABEL_1 -> AI),
    applied during the original screening because none of these checkpoints populate
    `id2label` in config.json.
  * "corrected"   - the convention specified by each publisher's reference
    implementation (see Supplementary Table S5 for provenance).

Only `MayZhou/e5-small-lora-ai-generated-detector` documents its convention on its
model card; that mapping matches the library default and its rate is identical under
both columns. For the other two the reference implementations invert it:

  openai-community/roberta-large-openai-detector
      github.com/openai/gpt-2-output-dataset -> detector/server.py:
          fake, real = probs.detach().cpu().flatten().numpy().tolist()
      => index 0 is machine-generated.

  TrustSafeAI/RADAR-Vicuna-7B
      huggingface.co/spaces/TrustSafeAI/RADAR-AI-Text-Detector -> app.py:
          output_probs = F.log_softmax(detector(**inputs).logits, -1)[:, 0].exp()
      => index 0 is AI-generated.

Binoculars is not covered here: it was identified as a candidate but never evaluated,
as its reference configuration (tiiuae/falcon-7b + falcon-7b-instruct) requires model
weights exceeding available hardware.

Usage:
    python rerun_excluded_detectors.py --input results/icnale.csv \
                                       --output results/excluded_detectors_icnale.csv

Input must be a per-essay CSV carrying `Country`, `Proficiency` and `Text` columns
(the schema emitted by run_detectors_on_icnale.py).
"""
import argparse

import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# (checkpoint, revision, index of the logit meaning "AI-generated" per the
#  publisher's reference implementation)
DETECTORS = [
    ("openai-community/roberta-large-openai-detector", "38f3e0ccf205", 0),
    ("MayZhou/e5-small-lora-ai-generated-detector",    "483fc4969592", 1),
    ("TrustSafeAI/RADAR-Vicuna-7B",                    "4ff1f23a69a3", 0),
]

THRESHOLD = 0.5
MAX_TOKENS = 512


def score(checkpoint, revision, ai_index, texts, device, batch_size):
    """Return P(AI) per text under the publisher's documented convention."""
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, revision=revision)
    model = AutoModelForSequenceClassification.from_pretrained(checkpoint, revision=revision)
    model.to(device).eval()

    probs = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        encoded = tokenizer(batch, truncation=True, max_length=MAX_TOKENS,
                            padding=True, return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**encoded).logits
        probs += torch.softmax(logits, dim=-1).cpu()[:, ai_index].tolist()

    del model
    return probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="per-essay ICNALE CSV")
    ap.add_argument("--output", required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--device", default=None, help="cuda / mps / cpu (auto-detected)")
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available()
                             else "mps" if torch.backends.mps.is_available() else "cpu")

    df = pd.read_csv(args.input).dropna(subset=["Text"])
    df["Text"] = df["Text"].astype(str)
    df = df.drop_duplicates(subset=["Text"]).reset_index(drop=True)
    df.insert(0, "essay_id", [f"ICNALE_{i + 1:04d}" for i in range(len(df))])
    print(f"{len(df)} human-authored essays | device={device}")

    for checkpoint, revision, ai_index in DETECTORS:
        name = checkpoint.split("/")[-1]
        print(f"scoring {name} (AI logit index {ai_index}) ...")
        p_ai = score(checkpoint, revision, ai_index, df["Text"].tolist(),
                     device, args.batch_size)

        df[f"{name} Score"] = p_ai
        df[f"{name} Label (corrected)"] = [
            "ChatGPT" if p > THRESHOLD else "Human" for p in p_ai
        ]
        # The original screening called it AI whenever the argmax fell on LABEL_1.
        df[f"{name} Label (as screened)"] = [
            "ChatGPT" if (p <= THRESHOLD if ai_index == 0 else p > THRESHOLD) else "Human"
            for p in p_ai
        ]

    df.drop(columns=["Text"]).to_csv(args.output, index=False)
    print(f"wrote {args.output}")

    print("\nMisclassification rate on human-authored text (n = %d)" % len(df))
    print(f"{'Detector':<40}{'as screened':>14}{'corrected':>12}")
    for checkpoint, _, _ in DETECTORS:
        name = checkpoint.split("/")[-1]
        old = (df[f"{name} Label (as screened)"] == "ChatGPT").mean()
        new = (df[f"{name} Label (corrected)"] == "ChatGPT").mean()
        print(f"{name:<40}{100 * old:>13.2f}%{100 * new:>11.2f}%")


if __name__ == "__main__":
    main()
