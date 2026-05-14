"""Compute the fairness metrics reported in the paper from a results CSV.

Given a CSV produced by run_detectors_on_*.py, this script reports:
  - False-positive (misclassification) rate per demographic group, per detector
  - Bias range (max - min FPR across groups) per detector
  - Cohen's d on a chosen linguistic feature, correctly classified vs misclassified
  - Separation Index (median diff / pooled IQR) on the same feature

Pass --group-col to pick the demographic axis (e.g. Country, Proficiency,
Academic_Field, Gender, "Age Group"). Multiple detectors are auto-detected
from columns named "<detector> Label".

Example:
  python compute_bias_metrics.py \\
      --input  results/icnale.csv \\
      --group-col Country \\
      --feature "Avg Word Length"
"""

import argparse
import re

import numpy as np
import pandas as pd


def detect_columns(df: pd.DataFrame):
    """Find every '<detector> Label' column and its companion 'Score' column."""
    detectors = []
    for col in df.columns:
        m = re.match(r"^(.+) Label$", col)
        if m and m.group(1) not in {"Golden", "Predicted"}:
            name = m.group(1)
            if f"{name} Score" in df.columns:
                detectors.append(name)
    return detectors


def fpr_by_group(df, detector, group_col):
    sub = df[df["Golden Label"] == "Human"].copy()
    sub = sub[sub[f"{detector} Label"] != "ERROR"]
    return (
        sub.groupby(group_col)[f"{detector} Label"]
           .apply(lambda s: 100 * (s == "ChatGPT").mean())
           .round(2)
    )


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    n1, n2 = len(a), len(b)
    s1, s2 = a.std(ddof=1), b.std(ddof=1)
    pooled = np.sqrt(((n1 - 1) * s1 ** 2 + (n2 - 1) * s2 ** 2) / (n1 + n2 - 2))
    if pooled == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled


def separation_index(a: pd.Series, b: pd.Series) -> float:
    a, b = a.dropna(), b.dropna()
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    iqr_a = a.quantile(0.75) - a.quantile(0.25)
    iqr_b = b.quantile(0.75) - b.quantile(0.25)
    pooled_iqr = (iqr_a + iqr_b) / 2
    if pooled_iqr == 0:
        return float("nan")
    return abs(a.median() - b.median()) / pooled_iqr


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--group-col", required=True,
                    help="Demographic column to slice on (e.g. Country, Proficiency)")
    ap.add_argument("--feature", default="Avg Word Length",
                    help="Linguistic feature column to use for Cohen's d / SI")
    ap.add_argument("--output", default=None, help="Optional CSV destination for FPR table")
    args = ap.parse_args()

    df = pd.read_csv(args.input)
    detectors = detect_columns(df)
    if not detectors:
        raise SystemExit("No '<detector> Label' columns detected — was this CSV produced by run_detectors_on_*.py?")
    print(f"Detectors found: {', '.join(detectors)}\n")

    # 1. FPR table
    fpr_table = pd.DataFrame({d: fpr_by_group(df, d, args.group_col) for d in detectors})
    fpr_table["Sample Size"] = df[df["Golden Label"] == "Human"].groupby(args.group_col).size()
    print(f"=== False-Positive Rate by {args.group_col} (%) ===")
    print(fpr_table.sort_values(detectors[0], ascending=False).to_string())

    # 2. Bias range
    print("\n=== Bias Range (max - min FPR) ===")
    for d in detectors:
        vals = fpr_table[d].dropna()
        if not vals.empty:
            print(f"  {d:<48s} {vals.max() - vals.min():6.2f} pp")

    # 3. Cohen's d and SI on the chosen feature
    if args.feature in df.columns:
        print(f"\n=== Linguistic-feature contrast: {args.feature} ===")
        print("(correctly classified vs misclassified, human-only)")
        human = df[df["Golden Label"] == "Human"]
        for d in detectors:
            correct = human[human[f"{d} Label"] == "Human"][args.feature]
            misclassified = human[human[f"{d} Label"] == "ChatGPT"][args.feature]
            d_val = cohens_d(correct, misclassified)
            si_val = separation_index(correct, misclassified)
            print(f"  {d:<48s} Cohen's d={d_val:+.2f}   Separation Index={si_val:.2f}   "
                  f"(n_correct={len(correct)}, n_misclassified={len(misclassified)})")
    else:
        print(f"\nFeature '{args.feature}' not found in CSV; skipping Cohen's d / SI.")

    if args.output:
        fpr_table.to_csv(args.output)
        print(f"\nFPR table saved to {args.output}")


if __name__ == "__main__":
    main()
