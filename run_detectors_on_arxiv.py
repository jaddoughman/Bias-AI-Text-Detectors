"""Run MGT detectors on arXiv abstracts.

Expects either:
  (a) a CSV with columns: id, categories, abstract  (Kaggle Cornell-University/arxiv export), OR
  (b) a JSONL with the same fields (one paper per line).

A balanced sample is drawn (90 abstracts per field by default, no minimum length) and
each abstract is passed to every detector listed in --detectors. Output is a
CSV with text + arXiv metadata + linguistic features + per-detector scores.

Example:
  python run_detectors_on_arxiv.py \\
      --input  arxiv-metadata-oai-snapshot.json \\
      --output results/arxiv.csv \\
      --per-field 90 --min-chars 0

The published study used 90 abstracts per field (990; the texts of two were not
retained, leaving 988) with no minimum length. GPTZero was queried on all of them;
ZeroGPT, Originality.AI and the 2025 Hello-SimpleAI/Fakespot runs used the first 45
per field. Earlier versions of this file defaulted to 200 per field and a 1,500-
character minimum, which did not match the study (see the manuscript, Section 3.1).
The exact sample (arXiv identifiers) is in outputs/arxiv_outputs.csv; to re-score it,
use score_local_detectors.py on those texts rather than re-sampling.
"""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from text_detector import TextDetector
from linguistic_features import LinguisticFeatureExtractor


# Map of arXiv primary-category prefixes to the 11 field labels used in §3.1.
CATEGORY_TO_FIELD = {
    "cs": "Computer Science",
    "math": "Mathematics",
    "astro-ph": "Astrophysics",
    "cond-mat": "Condensed Matter",
    "hep": "High Energy Physics",
    "stat": "Statistics",
    "q-bio": "Quantitative Biology",
    "q-fin": "Quantitative Finance",
    "eess": "Electrical Engineering",
    "econ": "Economics",
}


def classify_field(categories: str) -> str:
    """Map a space-separated category string to one of the 11 field labels."""
    if not isinstance(categories, str):
        return "Other"
    primary = categories.split()[0]
    for prefix, field in CATEGORY_TO_FIELD.items():
        if primary.startswith(prefix):
            return field
    return "Other"


def load_papers(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in {".jsonl", ".json"}:
        with p.open() as f:
            records = [json.loads(line) for line in f if line.strip()]
        df = pd.DataFrame(records)
    else:
        df = pd.read_csv(p)
    return df


DEFAULT_DETECTORS = [
    "www.originality.ai",
    "www.zerogpt.com",
    "Hello-SimpleAI/chatgpt-detector-roberta",
    "fakespot-ai/roberta-base-ai-text-detection-v1",
]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="arXiv metadata CSV or JSONL")
    ap.add_argument("--output", required=True, help="Destination CSV path")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--category-col", default="categories")
    ap.add_argument("--abstract-col", default="abstract")
    ap.add_argument("--per-field", type=int, default=90, help="Abstracts sampled per field")
    ap.add_argument("--min-chars", type=int, default=0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS)
    ap.add_argument("--api-delay", type=float, default=1.5)
    args = ap.parse_args()

    print(f"Loading {args.input} ...")
    df = load_papers(args.input)
    df = df.rename(columns={
        args.id_col: "id",
        args.category_col: "categories",
        args.abstract_col: "abstract",
    })
    df["abstract"] = df["abstract"].astype(str).str.strip()
    df = df[df["abstract"].str.len() >= args.min_chars].copy()
    df["Academic_Field"] = df["categories"].apply(classify_field)

    sampled = (
        df.groupby("Academic_Field", group_keys=False)
          .apply(lambda g: g.sample(n=min(args.per_field, len(g)), random_state=args.seed))
          .reset_index(drop=True)
    )
    print(f"Sampled {len(sampled)} abstracts across {sampled['Academic_Field'].nunique()} fields")

    detector = TextDetector()
    feat_extractor = LinguisticFeatureExtractor()

    rows = []
    for i, (_, paper) in enumerate(sampled.iterrows(), 1):
        text = paper["abstract"]
        row = {
            "ArXiv_ID": paper["id"],
            "Categories": paper["categories"],
            "Academic_Field": paper["Academic_Field"],
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
        if i % 25 == 0 or i == len(sampled):
            print(f"  [{i}/{len(sampled)}] {paper['Academic_Field']}")
            pd.DataFrame(rows).to_csv(args.output, index=False)

    pd.DataFrame(rows).to_csv(args.output, index=False)
    print(f"\nWrote {len(rows)} rows to {args.output}")


if __name__ == "__main__":
    main()
