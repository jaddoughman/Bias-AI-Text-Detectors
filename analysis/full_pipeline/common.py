"""Shared definitions for the R5 re-analysis."""
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

REGION = {
    'India': 'South Asia', 'Nepal': 'South Asia', 'Bangladesh': 'South Asia', 'Pakistan': 'South Asia',
    'Brunei': 'Southeast Asia', 'Vietnam': 'Southeast Asia', 'Cambodia': 'Southeast Asia', 'Myanmar': 'Southeast Asia',
    'Malaysia': 'Southeast Asia', 'Laos': 'Southeast Asia', 'Singapore': 'Southeast Asia', 'Indonesia': 'Southeast Asia',
    'Philippines': 'Southeast Asia', 'Thailand': 'Southeast Asia',
    'Korea': 'East Asia', 'Japan': 'East Asia', 'Hong Kong': 'East Asia', 'China': 'East Asia', 'Taiwan': 'East Asia',
    'United Arab Emirates': 'Middle East', 'English Native Speakers': 'Native English',
}
RETAINED = ['originality', 'zerogpt', 'gptzero', 'hello', 'fakespot']
ADDED = ['roberta_large', 'radar']
NAMES = {'originality': 'Originality.AI', 'zerogpt': 'ZeroGPT', 'gptzero': 'GPTZero', 'hello': 'Hello-SimpleAI',
         'fakespot': 'Fakespot', 'roberta_large': 'RoBERTa-large (OpenAI)', 'radar': 'RADAR', 'e5': 'e5-small-lora'}


def wilson(k, n, alpha=0.05):
    if n == 0:
        return (np.nan, np.nan)
    lo, hi = proportion_confint(k, n, alpha=alpha, method='wilson')
    return lo, hi


def rate_table(df, group, dets, flag='{}_flag'):
    rows = []
    for g, sub in df.groupby(group, sort=False):
        r = {group: g}
        for d in dets:
            s = sub[flag.format(d)].dropna()
            k, n = int(s.sum()), int(len(s))
            lo, hi = wilson(k, n)
            r.update({f'{d}_k': k, f'{d}_n': n, f'{d}_rate': 100 * k / n if n else np.nan,
                      f'{d}_lo': 100 * lo, f'{d}_hi': 100 * hi})
        rows.append(r)
    return pd.DataFrame(rows)
