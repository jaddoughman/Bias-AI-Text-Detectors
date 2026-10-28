"""Recompute the main R5 tables from the per-text outputs in ../outputs/.

  python reproduce_tables.py            # writes ./tables_from_outputs/*.csv and prints key figures

FPR = flagged / scored on each detector's scored texts; 95% CIs are Wilson score intervals.
The conditional analyses (Section 4.4.3, S10 Table) use linear probability models with
country fixed effects and 1,000 bootstrap resamples stratified by country.
"""
import os

import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportion_confint

HERE = os.path.dirname(os.path.abspath(__file__))
OUTP = os.path.join(HERE, '..', 'outputs')
TAB = os.path.join(HERE, 'tables_from_outputs')
DETS = ['originality', 'zerogpt', 'gptzero', 'hello', 'fakespot', 'roberta_large', 'radar', 'e5']
REGION = {'India': 'South Asia', 'Nepal': 'South Asia', 'Bangladesh': 'South Asia', 'Pakistan': 'South Asia',
          'Brunei': 'Southeast Asia', 'Vietnam': 'Southeast Asia', 'Cambodia': 'Southeast Asia', 'Myanmar': 'Southeast Asia',
          'Malaysia': 'Southeast Asia', 'Laos': 'Southeast Asia', 'Singapore': 'Southeast Asia', 'Indonesia': 'Southeast Asia',
          'Philippines': 'Southeast Asia', 'Thailand': 'Southeast Asia', 'Korea': 'East Asia', 'Japan': 'East Asia',
          'Hong Kong': 'East Asia', 'China': 'East Asia', 'Taiwan': 'East Asia', 'United Arab Emirates': 'Middle East',
          'Native English': 'Native English'}
FEATS = ['Avg Word Length', 'Flesch Reading Ease', 'Lexical Diversity', 'Paragraph Count', 'Word Count']


def rate(s):
    s = s.dropna()
    k, n = int(s.sum()), len(s)
    lo, hi = proportion_confint(k, n, method='wilson') if n else (np.nan, np.nan)
    return pd.Series({'k': k, 'n': n, 'fpr': 100 * k / n if n else np.nan, 'lo': 100 * lo, 'hi': 100 * hi})


def table(df, col, dets=DETS):
    out = []
    for d in dets:
        t = df.groupby(col)[f'{d}_flag'].apply(rate).unstack()
        t.insert(0, 'detector', d)
        out.append(t.reset_index())
    return pd.concat(out)


def lpm_gap(s, group, base, target, fe=True, feats=False):
    X = [np.ones(len(s))]
    names = ['const']
    for lev in sorted(set(s[group]) - {base}):
        X.append((s[group].values == lev).astype(float)); names.append(lev)
    if fe:
        for c in sorted(set(s.Country))[1:]:
            X.append((s.Country.values == c).astype(float)); names.append('C=' + c)
    if feats:
        for f in FEATS:
            X.append(((s[f] - s[f].mean()) / s[f].std()).values); names.append(f)
    beta, *_ = np.linalg.lstsq(np.column_stack(X), s.y.values, rcond=None)
    return beta[names.index(target)]


def conditional(ic, d, B=1000, seed=7):
    s = ic[ic[f'{d}_flag'].notna() & (ic.Proficiency != 'Native')].copy()
    s['y'] = s[f'{d}_flag'].astype(float)

    def stats(x):
        r = x[x.Region.isin(['South Asia', 'East Asia', 'Southeast Asia', 'Middle East'])]
        return (lpm_gap(x, 'Proficiency', 'A2', 'B2+'), lpm_gap(x, 'Proficiency', 'A2', 'B2+', feats=True),
                lpm_gap(r, 'Region', 'East Asia', 'South Asia', fe=False), lpm_gap(r, 'Region', 'East Asia', 'South Asia', fe=False, feats=True))
    est = stats(s)
    rng = np.random.default_rng(seed)
    groups = [g.index.values for _, g in s.groupby('Country')]
    bs = np.array([stats(s.loc[np.concatenate([rng.choice(g, len(g)) for g in groups])].reset_index(drop=True)) for _ in range(B)])
    lo, hi = np.percentile(bs, 2.5, axis=0), np.percentile(bs, 97.5, axis=0)
    names = ['b2_minus_a2_within_country', 'b2_minus_a2_within_country_plus_features', 'south_minus_east_asia', 'south_minus_east_asia_plus_features']
    return pd.DataFrame({'detector': d, 'quantity': names, 'pp': 100 * np.array(est), 'lo': 100 * lo, 'hi': 100 * hi})


def main():
    os.makedirs(TAB, exist_ok=True)
    ic = pd.read_csv(os.path.join(OUTP, 'icnale_outputs.csv'))
    ic['Region'] = ic.Country.map(REGION)
    bl = pd.read_csv(os.path.join(OUTP, 'blog_outputs.csv'))
    ax = pd.read_csv(os.path.join(OUTP, 'arxiv_outputs.csv'))
    table(ic, 'Country').to_csv(os.path.join(TAB, 'icnale_country.csv'), index=False)
    table(ic, 'Region').to_csv(os.path.join(TAB, 'icnale_region.csv'), index=False)
    table(ic, 'Proficiency').to_csv(os.path.join(TAB, 'icnale_cefr.csv'), index=False)
    ib = ic[(ic.Country == 'India') & (ic.Proficiency == 'B2+')].assign(cell='India B2+')
    table(ib, 'cell').to_csv(os.path.join(TAB, 'india_b2.csv'), index=False)
    table(bl, 'Gender').to_csv(os.path.join(TAB, 'blog_gender.csv'), index=False)
    table(bl, 'Age Group').to_csv(os.path.join(TAB, 'blog_age.csv'), index=False)
    table(ax, 'Academic_Field').to_csv(os.path.join(TAB, 'arxiv_field.csv'), index=False)
    cond = pd.concat([conditional(ic, d) for d in DETS[:7]])
    cond.to_csv(os.path.join(TAB, 'conditional.csv'), index=False)
    c = table(ic, 'Country')
    print('Country FPR range (pp):', c.groupby('detector').fpr.agg(lambda x: round(x.max() - x.min(), 1)).to_dict())
    print(cond.round(1).to_string(index=False))


if __name__ == '__main__':
    main()
