"""Run MGT detectors on AI-generated essays from DAIGT v2 to compute TPR.

Source: https://www.kaggle.com/datasets/thedrcat/daigt-v2-train-dataset
We sample 100 essays from each of five generators (configurable) under the
same protocol used in the paper's §4.5 (random_state=42, ≥200 chars, 0.5
classification threshold).

Outputs a CSV with one row per AI essay containing the source generator,
prompt, the text itself, and (label, score) for every detector.

Example:
  python run_detectors_on_daigt.py \\
      --input  train_v2_drcat_02.csv \\
      --output results/daigt_ai.csv \\
      --per-generator 100
"""

import argparse
import time

import pandas as pd

from text_detector import TextDetector
from linguistic_features import LinguisticFeatureExtractor


DEFAULT_GENERATORS = [
    "mistral7binstruct_v1",
    "mistral7binstruct_v2",
    "chat_gpt_moth",
    "llama2_chat",
    "kingki19_palm",
]

# Quillbot/GPTZero omitted from default for the same reasons as the
# human-corpus scripts. Add them via --detectors if you have credentials.
DEFAULT_DETECTORS = [
    "www.originality.ai",
    "Hello-SimpleAI/chatgpt-detector-roberta",
    "fakespot-ai/roberta-base-ai-text-detection-v1",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="DAIGT v2 CSV (train_v2_drcat_02.csv)")
    ap.add_argument("--output", required=True, help="Destination CSV path")
    ap.add_argument("--per-generator", type=int, default=100)
    ap.add_argument("--generators", nargs="+", default=DEFAULT_GENERATORS)
    ap.add_argument("--min-chars", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS)
    ap.add_argument("--api-delay", type=float, default=1.5)
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df = df[df["label"] == 1].copy()  # AI-generated only
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() >= args.min_chars]

    pool = []
    for gen in args.generators:
        sub = df[df["source"] == gen]
        if sub.empty:
            print(f"  WARNING: generator '{gen}' has no rows after filtering, skipping")
            continue
        sampled = sub.sample(n=min(args.per_generator, len(sub)), random_state=args.seed)
        pool.extend(sampled.to_dict("records"))
        print(f"  {gen:<28s} sampled {len(sampled)}")
    print(f"\nTotal pool: {len(pool)} essays\n")

    detector = TextDetector()
    feat_extractor = LinguisticFeatureExtractor()

    rows = []
    for i, rec in enumerate(pool, 1):
        text = rec["text"]
        row = {
            "Generator": rec["source"],
            "Prompt": rec.get("prompt_name", ""),
            "Char_Length": len(text),
            "Text": text,
            "Golden Label": "ChatGPT",  # AI-generated
        }
        row.update(feat_extractor.extract_features(text).to_dict())

        for det in args.detectors:
            try:
                label, score = detector.classify(text, det)
                row[f"{det} Label"] = label
                row[f"{det} Score"] = score
                row[f"{det} Correct"] = (label == "ChatGPT")
            except Exception as e:
                row[f"{det} Label"] = "ERROR"
                row[f"{det} Score"] = None
                row[f"{det} Correct"] = False
                row[f"{det} Error"] = str(e)[:200]
            if det.startswith("www.") or det == "gptzero":
                time.sleep(args.api_delay)

        rows.append(row)
        if i % 25 == 0 or i == len(pool):
            print(f"  [{i}/{len(pool)}] {rec['source']}")
            pd.DataFrame(rows).to_csv(args.output, index=False)

    pd.DataFrame(rows).to_csv(args.output, index=False)

    # Print a TPR summary so the user can see results without a second script.
    print("\n=== True-Positive Rate by Generator ===")
    out = pd.DataFrame(rows)
    for gen in args.generators:
        sub = out[out["Generator"] == gen]
        if sub.empty:
            continue
        line = [f"{gen:<28s}"]
        for det in args.detectors:
            col = f"{det} Correct"
            valid = sub[sub[f"{det} Label"] != "ERROR"]
            tpr = 100 * valid[col].sum() / len(valid) if len(valid) else float("nan")
            line.append(f"{det.split('/')[-1][:20]}={tpr:5.1f}%")
        print(" | ".join(line))

    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
