"""Run MGT detectors on the Blog Authorship Corpus.

Expects a CSV with at least these columns (default names match the Kaggle
`rtatman/blog-authorship-corpus` distribution):
  id, gender, age, topic, sign, date, text

Outputs a CSV with text + demographic metadata + linguistic features +
per-detector (label, score) columns.

Example:
  python run_detectors_on_blogs.py \\
      --input  blogtext.csv \\
      --output results/blogs.csv \\
      --sample 2000 --balance-by gender --min-chars 501

The published study used 2,000 posts longer than 500 characters, 1,000 per gender,
drawn at random within gender (age follows the corpus distribution; it was not
balanced). The exact sample (post texts with gender and age) is in
outputs/blog_outputs.csv; to re-score it, use score_local_detectors.py on those texts
rather than re-sampling.
"""

import argparse
import time

import pandas as pd

from text_detector import TextDetector
from linguistic_features import LinguisticFeatureExtractor


DEFAULT_DETECTORS = [
    "www.originality.ai",
    "www.zerogpt.com",
    "Hello-SimpleAI/chatgpt-detector-roberta",
    "fakespot-ai/roberta-base-ai-text-detection-v1",
]


def age_to_group(age):
    try:
        age = int(age)
    except (TypeError, ValueError):
        return "Unknown"
    if 13 <= age <= 17:
        return "Adolescent (13-17)"
    if 23 <= age <= 27:
        return "Young Adult (23-27)"
    if 33 <= age <= 47:
        return "Adult (33-47)"
    return "Other (48+)"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Blog corpus CSV path")
    ap.add_argument("--output", required=True, help="Destination CSV path")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--gender-col", default="gender")
    ap.add_argument("--age-col", default="age")
    ap.add_argument("--sample", type=int, default=None, help="Optional row cap")
    ap.add_argument("--balance-by", nargs="+", default=None,
                    help="Column(s) to stratify the sample by (e.g. 'gender age_group')")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-chars", type=int, default=501,
                    help="Skip posts shorter than this")
    ap.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS)
    ap.add_argument("--api-delay", type=float, default=1.5)
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input)
    df = df.rename(columns={
        args.text_col: "text", args.gender_col: "gender", args.age_col: "age",
    })
    df["text"] = df["text"].astype(str)
    df = df[df["text"].str.len() >= args.min_chars].copy()
    df["age_group"] = df["age"].apply(age_to_group)

    if args.sample:
        if args.balance_by:
            grouped = df.groupby(args.balance_by, group_keys=False)
            per_cell = max(1, args.sample // grouped.ngroups)
            df = grouped.apply(
                lambda g: g.sample(n=min(per_cell, len(g)), random_state=args.seed)
            ).reset_index(drop=True)
        else:
            df = df.sample(n=min(args.sample, len(df)), random_state=args.seed)
    print(f"Evaluating {len(df)} posts")

    detector = TextDetector()
    feat_extractor = LinguisticFeatureExtractor()

    rows = []
    for i, (_, post) in enumerate(df.iterrows(), 1):
        text = post["text"]
        row = {
            "Gender": post["gender"],
            "Age": post["age"],
            "Age Group": post["age_group"],
            "Text": text,
            "Golden Label": "Human",
        }
        row.update(feat_extractor.extract_features(text).to_dict())

        for det in args.detectors:
            try:
                label, score = detector.classify(text, det)
                row[f"{det} Label"] = label
                row[f"{det} Score"] = score
            except Exception as e:
                row[f"{det} Label"] = "ERROR"
                row[f"{det} Score"] = None
                row[f"{det} Error"] = str(e)[:200]
            if det.startswith("www.") or det == "gptzero":
                time.sleep(args.api_delay)

        rows.append(row)
        if i % 25 == 0 or i == len(df):
            print(f"  [{i}/{len(df)}]")
            pd.DataFrame(rows).to_csv(args.output, index=False)

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
