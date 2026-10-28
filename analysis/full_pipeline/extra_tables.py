"""GPTZero discarded-output sensitivity, prompt-effect check, and per-detector coverage reasons."""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import NAMES

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
TAB = os.path.join(R, 'tables')
MAIN = ['originality', 'zerogpt', 'gptzero', 'hello', 'fakespot', 'roberta_large', 'radar']


def gptzero_dropped():
    m = pd.read_csv(os.path.join(R, 'data', 'icnale_final_noText.csv'))
    g = m[m.gptzero_in_sample].copy()
    g['upper'] = g.gptzero_flag.fillna(1)
    rows = []
    for c, s in g.groupby('Country'):
        k, n = int(s.gptzero_flag.sum()), int(s.gptzero_flag.notna().sum())
        rows.append(dict(Country=c, dropped=len(s) - n, n_scored=n, k=k, rate_published=100 * k / n,
                         k_if_flag=int(s.upper.sum()), n_all=len(s), rate_if_flag=100 * s.upper.mean()))
    t = pd.DataFrame(rows).sort_values(['rate_if_flag', 'rate_published'], ascending=False)
    t.to_csv(os.path.join(TAB, 'gptzero_dropped.csv'), index=False)
    dr, kept = g[g.gptzero_flag.isna()], g[g.gptzero_flag.notna()]
    parts = []
    for d in ['originality', 'zerogpt', 'hello', 'fakespot', 'roberta_large', 'radar']:
        a, b = dr[f'{d}_flag'].dropna(), kept[f'{d}_flag'].dropna()
        parts.append(f'{NAMES[d]} {100 * a.mean():.0f}\\% ({int(a.sum())}/{len(a)}) vs {100 * b.mean():.0f}\\%')
    note = ('Share of the 24 discarded essays flagged by the other detectors, versus the share of retained essays: '
            + '; '.join(parts) + '. By CEFR level the discarded essays were B1\\_1 (10), B1\\_2 (10) and B2+ (4); '
            'none were A2 or native-speaker essays.')
    lv = dr.Proficiency.value_counts().to_dict()
    note = note.replace('B1\\_1 (10), B1\\_2 (10) and B2+ (4)',
                        f"B1\\_1 ({lv.get('B1_1', 0)}), B1\\_2 ({lv.get('B1_2', 0)}) and B2+ ({lv.get('B2+', 0)})")
    open(os.path.join(TAB, 'gptzero_dropped_note.txt'), 'w').write(note)
    # Blog / arXiv GPTZero subsets
    out = dict(icnale=dict(sample=len(g), scored=int(g.gptzero_flag.notna().sum())))
    json.dump(out, open(os.path.join(TAB, 'gptzero_dropped_summary.json'), 'w'))


def prompt_check():
    m = pd.read_csv(os.path.join(R, 'data', 'icnale_final_noText.csv'))
    rows = []
    for d in MAIN:
        s = m[m[f'{d}_flag'].notna()].copy()
        s['y'] = s[f'{d}_flag']
        rng = {}
        for p in ['PTJ', 'SMK']:
            t = s[s.prompt == p].groupby('Country').y.mean() * 100
            rng[p] = t.max() - t.min()
        r2c = smf.ols('y ~ C(Country)', data=s).fit().rsquared
        r2p = smf.ols('y ~ C(prompt)', data=s).fit().rsquared
        fit = smf.ols('y ~ C(Country) + C(prompt)', data=s).fit()
        pr = s.groupby('prompt').y.mean() * 100
        rows.append(dict(detector=NAMES[d], range_ptj=rng['PTJ'], range_smk=rng['SMK'], fpr_ptj=pr['PTJ'], fpr_smk=pr['SMK'],
                         r2_country=r2c, r2_prompt=r2p, prompt_coef=100 * fit.params['C(prompt)[T.SMK]'],
                         prompt_p=fit.pvalues['C(prompt)[T.SMK]']))
    pd.DataFrame(rows).to_csv(os.path.join(TAB, 'prompt_check.csv'), index=False)


def coverage_reasons():
    rs = {
        'ICNALE|Originality.AI': 'Scores returned for 672 essays (API-side throttling of the free endpoint during collection); evenly spread across countries (20--40\\% per country) and CEFR levels (30--36\\%).',
        'ICNALE|ZeroGPT': 'Scores returned for 1{,}991 essays (API-side throttling); 91--100\\% coverage per country.',
        'ICNALE|GPTZero': 'Queried on the first 50 sampled essays per country (1{,}050); 24 outputs other than human/AI (``mixed'' or failed) were discarded by the collection script (S8 Table).',
        'Blog|Originality.AI': 'Scores returned for 1{,}251 posts (626 male, 625 female); age distribution within 1 pp of the full sample.',
        'Blog|ZeroGPT': 'Three API errors (previously counted as flagged; now treated as unscored).',
        'Blog|GPTZero': 'Queried on a gender-balanced subsample of 1{,}000 posts; 2 outputs other than human/AI discarded.',
        'arXiv|Originality.AI': 'Queried on the 45-per-field subsample (495 abstracts); scores returned for 356 (API-side throttling).',
        'arXiv|ZeroGPT': 'Queried on a 45-per-field subsample of the 90-per-field sample (495 abstracts).',
        'arXiv|GPTZero': 'Queried on the full sample; 4 outputs other than human/AI discarded (2 of these abstracts are not recoverable and are absent from all analyses).',
    }
    json.dump(rs, open(os.path.join(TAB, 'coverage_reasons.json'), 'w'), indent=1)


if __name__ == '__main__':
    gptzero_dropped(); prompt_check(); coverage_reasons()
    print(open(os.path.join(TAB, 'gptzero_dropped_note.txt')).read())
    print(pd.read_csv(os.path.join(TAB, 'prompt_check.csv')).round(3).to_string())
