"""Run the conditional analyses (Section 4.4.3, S10 Table) for all seven analysed detectors.

Reads revision03/data/icnale_final_noText.csv (written by tables_r5.py).
Writes tables/conditional.csv, tables/conditional_matched.csv, tables/intersection_check.csv,
tables/cohens_d.csv, tables/linguistic_profiles.csv, tables/whitespace.csv.
"""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from common import NAMES, wilson
from conditional import FEATURES, boot, matched, penalty, prep, region_gap, strata

warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
TAB = os.path.join(R, 'tables')
MAIN = ['originality', 'zerogpt', 'gptzero', 'hello', 'fakespot', 'roberta_large', 'radar']
ALLF = ['Word Count', 'Sentence Count', 'Avg Word Length', 'Lexical Diversity', 'Avg Sentence Length',
        'Flesch Reading Ease', 'Flesch-Kincaid Grade Level', 'Gunning Fog Index', 'Paragraph Count',
        'Avg Sentences per Paragraph', 'Avg Words per Paragraph', 'Avg Paragraph Length']


def cohens_d(a, b):
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var() + (nb - 1) * b.var()) / (na + nb - 2))
    return (b.mean() - a.mean()) / sp if sp > 0 else np.nan


# Countries with at least four misclassified essays on Originality.AI, plus four zero-FPR comparison countries
PROFILE_COUNTRIES = ['Brunei', 'Vietnam', 'India', 'Bangladesh', 'Nepal', 'Cambodia', 'Myanmar', 'Malaysia',
                     'Indonesia', 'Thailand', 'Japan', 'China']


def profiles(m):
    """Country linguistic profiles (Table 15) for Originality.AI and ZeroGPT."""
    prow = []
    for d in ['originality', 'zerogpt']:
        for c in PROFILE_COUNTRIES:
            t = m[(m.Country == c) & m[f'{d}_flag'].notna()]
            a, b = t[t[f'{d}_flag'] == 0], t[t[f'{d}_flag'] == 1]
            prow.append(dict(detector=NAMES[d], country=c, n_scored=len(t), n_correct=len(a), n_misclassified=len(b),
                             fpr=100 * len(b) / len(t), awl_correct=a['Avg Word Length'].mean(),
                             awl_mis=b['Avg Word Length'].mean() if len(b) else np.nan,
                             fre_correct=a['Flesch Reading Ease'].mean(),
                             fre_mis=b['Flesch Reading Ease'].mean() if len(b) else np.nan))
    pd.DataFrame(prow).to_csv(os.path.join(TAB, 'linguistic_profiles.csv'), index=False)


def main(B=1000):
    m = pd.read_csv(os.path.join(R, 'data', 'icnale_final_noText.csv'))
    m['Country'] = m.Country.replace({'UAE': 'United Arab Emirates'})
    l2 = m[m.Proficiency != 'Native']
    q_w, q_f = strata(l2)

    rows, mrows, irows, drows = [], [], [], []
    for d in MAIN:
        s = prep(m, d)
        s['Region'] = s.Region
        g0, g1 = penalty(s)
        bs = boot(s, penalty, B=B)
        r0, r1 = region_gap(s)
        br = boot(s, region_gap, B=B)
        share = 1 - g1 / g0 if g0 > 0 else np.nan
        bshare = 1 - bs[:, 1] / bs[:, 0]
        # marginal (no country FE) B2+ - A2 on the same essays
        marg = s[s.Proficiency == 'B2+'].y.mean() - s[s.Proficiency == 'A2'].y.mean()
        rows.append(dict(detector=NAMES[d], n=len(s), marginal_b2_a2=100 * marg,
                         within_country_b2_a2=100 * g0, lo0=100 * np.percentile(bs[:, 0], 2.5), hi0=100 * np.percentile(bs[:, 0], 97.5),
                         within_country_b2_a2_features=100 * g1, lo1=100 * np.percentile(bs[:, 1], 2.5), hi1=100 * np.percentile(bs[:, 1], 97.5),
                         share_accounted=100 * share if g0 > 0.02 else np.nan,
                         sa_ea_gap=100 * r0, sa_lo0=100 * np.percentile(br[:, 0], 2.5), sa_hi0=100 * np.percentile(br[:, 0], 97.5),
                         sa_ea_gap_features=100 * r1, sa_lo1=100 * np.percentile(br[:, 1], 2.5), sa_hi1=100 * np.percentile(br[:, 1], 97.5)))
        # high-risk region comparison
        hr = (s['Avg Word Length'] >= 4.7) & (s['Flesch Reading Ease'] <= 55)
        for reg in ['South Asia', 'Southeast Asia', 'East Asia']:
            sub = s[hr & (s.Region == reg)]
            k, n = int(sub.y.sum()), len(sub)
            lo, hi = wilson(k, n) if n else (np.nan, np.nan)
            mrows.append(dict(detector=NAMES[d], region=reg, share_in_high_risk=100 * hr[s.Region == reg].mean(),
                              n_high_risk=n, flagged=k, fpr=100 * k / n if n else np.nan, lo=100 * lo, hi=100 * hi,
                              fpr_outside=100 * s[(~hr) & (s.Region == reg)].y.mean()))
        mm = matched(s, q_w, q_f, 'Region', 'South Asia', 'East Asia')
        mrows.append(dict(detector=NAMES[d], region='SA vs EA, quintile-matched (std. to SA)',
                          share_in_high_risk=np.nan, n_high_risk=mm['n_b'], flagged=np.nan,
                          fpr=100 * mm['std_b'], lo=np.nan, hi=np.nan, fpr_outside=100 * mm['std_a']))
        # intersection check (penalised logit, additive main effects)
        X = pd.get_dummies(s[['Country', 'Proficiency']], drop_first=True).astype(float)
        lr = LogisticRegression(C=10.0, max_iter=2000).fit(X, s.y)
        cell = (s.Country == 'India') & (s.Proficiency == 'B2+')
        pred = lr.predict_proba(X[cell])[:, 1].mean()
        k, n = int(s.y[cell].sum()), int(cell.sum())
        from scipy.stats import binomtest
        p = binomtest(k, n, pred, alternative='greater').pvalue if n else np.nan
        irows.append(dict(detector=NAMES[d], n=n, flagged=k, observed=100 * k / n if n else np.nan,
                          predicted_additive=100 * pred, excess_pp=100 * (k / n - pred) if n else np.nan, p_one_sided=p))
        # Cohen's d (misclassified minus correct), all 12 features, all ICNALE essays scored
        t = m[m[f'{d}_flag'].notna()]
        a, b = t[t[f'{d}_flag'] == 0], t[t[f'{d}_flag'] == 1]
        r = dict(detector=NAMES[d], n_correct=len(a), n_misclassified=len(b))
        for f in ALLF:
            r[f] = cohens_d(a[f].dropna(), b[f].dropna())
        drows.append(r)

    pd.DataFrame(rows).to_csv(os.path.join(TAB, 'conditional.csv'), index=False)
    pd.DataFrame(mrows).to_csv(os.path.join(TAB, 'conditional_matched.csv'), index=False)
    pd.DataFrame(irows).to_csv(os.path.join(TAB, 'intersection_check.csv'), index=False)
    pd.DataFrame(drows).to_csv(os.path.join(TAB, 'cohens_d.csv'), index=False)

    profiles(m)

    # Whitespace normalisation experiment (local detectors)
    wsf = os.path.join(R, 'runs', 'icnale_ws_standard.csv')
    if os.path.exists(wsf):
        ws = pd.read_csv(wsf)
        assert (ws.text_id.values == m.text_id.values).all()
        nl = pd.read_csv(os.path.join(R, 'data', 'icnale_master.csv')).Text.str.contains('\n').values
        wr = []
        for d in ['hello', 'fakespot', 'roberta_large', 'radar']:
            orig = m[f'{d}_flag'].values
            new = ws[f'{d}_pred_ai'].astype(float).values
            for grp, mask in (('essays with line breaks', nl), ('essays without line breaks', ~nl)):
                wr.append(dict(detector=NAMES[d], group=grp, n=int(mask.sum()), fpr_original=100 * orig[mask].mean(),
                               fpr_whitespace_normalised=100 * new[mask].mean(),
                               flips_to_human=int(((orig == 1) & (new == 0) & mask).sum()),
                               flips_to_ai=int(((orig == 0) & (new == 1) & mask).sum())))
            cc = pd.DataFrame({'Country': m.Country, 'o': orig, 'n': new}).groupby('Country')[['o', 'n']].mean() * 100
            wr.append(dict(detector=NAMES[d], group='country range (pp)', n=len(m), fpr_original=cc.o.max() - cc.o.min(),
                           fpr_whitespace_normalised=cc.n.max() - cc.n.min(), flips_to_human=np.nan, flips_to_ai=np.nan))
        pd.DataFrame(wr).to_csv(os.path.join(TAB, 'whitespace.csv'), index=False)
    print(pd.DataFrame(rows).round(1).to_string())
    print(pd.DataFrame(irows).round(3).to_string())


if __name__ == '__main__':
    main()
