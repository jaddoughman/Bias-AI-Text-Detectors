"""Reconstruct the ICNALE evaluation sample exactly as drawn in bias_eval.ipynb.

Loader: ICNALE_WEP_0.5/WEP_1_Classified_Unmerged/<CC>/**/*.txt, essays > 500 chars,
up to N per country with pandas .sample(random_state=42). Because numpy's
RandomState.choice(replace=False) takes a prefix of one permutation, the
50-per-country GPTZero sample is the first 50 rows of the 100-per-country sample.
"""
import glob, os
import pandas as pd

COUNTRY = {'BGD':'Bangladesh','BRN':'Brunei','CHN':'China','ENS':'English Native Speakers','HKG':'Hong Kong',
 'IDN':'Indonesia','IND':'India','JPN':'Japan','KHM':'Cambodia','KOR':'Korea','LAO':'Laos','MMR':'Myanmar',
 'MYS':'Malaysia','NPL':'Nepal','PAK':'Pakistan','PHL':'Philippines','SIN':'Singapore','THA':'Thailand',
 'TWN':'Taiwan','UAE':'United Arab Emirates','VNM':'Vietnam'}
BASE = os.path.join(os.path.dirname(__file__), '..', '..', 'ICNALE_WEP_0.5', 'WEP_1_Classified_Unmerged')


def proficiency(fn):
    for tag, lab in (('A2','A2'),('B1_1','B1_1'),('B1_2','B1_2'),('B2','B2+')):
        if tag in fn:
            return lab
    return 'Unknown'


def load(order='sorted'):
    rows = []
    for cc, name in COUNTRY.items():
        files = glob.glob(os.path.join(BASE, cc, '**/*.txt'), recursive=True)
        if order == 'sorted':
            files = sorted(files)
        for fp in files:
            with open(fp, encoding='utf-8') as f:
                t = f.read().strip()
            if t:
                fn = os.path.basename(fp)
                rows.append(dict(text=t, country_code=cc, country=name, proficiency=proficiency(fn), filename=fn))
    return pd.DataFrame(rows)


def sample(df, n_per_country, seed=42):
    df = df[df.text.str.len() > 500]
    parts = [g.sample(n=min(n_per_country, len(g)), random_state=seed)
             for _, g in ((c, df[df.country == c]) for c in df.country.unique())]
    return pd.concat(parts).reset_index(drop=True)
