"""Compute every table reported in revision R5 from per-text detector outputs.

Inputs : revision03/data/{icnale,blog,arxiv}_master.csv  (published outputs, per text)
         revision03/runs/{icnale,blog,arxiv,daigt,dactyl}_standard.csv  (local re-runs)
         revision03/runs/{icnale,blog,arxiv}_original.csv  (replication of the 2025 wrapper)
Outputs: revision03/tables/*.csv and revision03/data/*_final.csv (per-text, all detectors)

Conventions
  * FPR = flagged / scored, computed on each detector's scored texts.
  * 95% CIs are Wilson score intervals (statsmodels proportion_confint, method='wilson').
  * Commercial detectors: September 2025 outputs as published (texts with an API error or a
    dropped GPTZero output are unscored, not counted as flagged).
  * Locally run detectors (Hello-SimpleAI, Fakespot, RoBERTa-large, RADAR, e5): re-run on every
    text with standard 512-token truncation at the pinned revisions in S5 Table.
"""
import json
import os

import numpy as np
import pandas as pd

from common import REGION, NAMES, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
TAB = os.path.join(R, 'tables')
os.makedirs(TAB, exist_ok=True)

COMMERCIAL = ['originality', 'zerogpt', 'gptzero']
LOCAL = ['hello', 'fakespot', 'roberta_large', 'radar']
MAIN = COMMERCIAL + LOCAL
ALL8 = MAIN + ['e5']


def load(ds):
    m = pd.read_csv(os.path.join(R, 'data', f'{ds}_master.csv'))
    run = pd.read_csv(os.path.join(R, 'runs', f'{ds}_standard.csv'))
    assert (run.text_id.values == m.text_id.values).all()
    for d in ['hello', 'fakespot']:
        m[f'{d}_pub_flag'] = m[f'{d}_flag']
        m[f'{d}_pub_score'] = m[f'{d}_score']
    for d in LOCAL + ['e5']:
        m[f'{d}_flag'] = run[f'{d}_pred_ai'].astype(float)
        m[f'{d}_p_ai'] = run[f'{d}_p_ai']
        m[f'{d}_n_tokens'] = run[f'{d}_n_tokens']
    orig = os.path.join(R, 'runs', f'{ds}_original.csv')
    if os.path.exists(orig):
        o = pd.read_csv(orig)
        for d in ['hello', 'fakespot']:
            m[f'{d}_orig_flag'] = o[f'{d}_pred_ai'].map({True: 1.0, False: 0.0, 'True': 1.0, 'False': 0.0})
            m[f'{d}_orig_error'] = o[f'{d}_error'].fillna('')
    # Commercial: API errors are unscored (published blog ZeroGPT analysis counted 3 as flags).
    for d in COMMERCIAL:
        m.loc[~m[f'{d}_label'].isin(['ChatGPT', 'Human']), f'{d}_flag'] = np.nan
    if ds == 'icnale':
        m['Region'] = m.Country.map(REGION)
        m['Country'] = m.Country.replace({'English Native Speakers': 'Native English', 'United Arab Emirates': 'UAE'})
    return m


def cell(s):
    s = s.dropna()
    k, n = int(s.sum()), int(len(s))
    lo, hi = wilson(k, n)
    return dict(k=k, n=n, rate=100 * k / n if n else np.nan, lo=100 * lo, hi=100 * hi)


def by_group(df, col, dets, order=None):
    rows = []
    groups = order if order is not None else list(dict.fromkeys(df[col]))
    for g in groups:
        sub = df[df[col] == g]
        r = {col: g}
        for d in dets:
            c = cell(sub[f'{d}_flag'])
            r.update({f'{d}_{k}': v for k, v in c.items()})
        rows.append(r)
    return pd.DataFrame(rows)


def ranges(t, dets, exclude=None):
    out = {}
    for d in dets:
        s = t if exclude is None else t[~t.iloc[:, 0].isin(exclude)]
        out[d] = s[f'{d}_rate'].max() - s[f'{d}_rate'].min()
    return out


def main():
    res = {}
    ic, bl, ax = load('icnale'), load('blog'), load('arxiv')
    for name, df in (('icnale', ic), ('blog', bl), ('arxiv', ax)):
        df.drop(columns=['Text']).to_csv(os.path.join(R, 'data', f'{name}_final_noText.csv'), index=False)

    # ---------- replication checks (published vs original-wrapper re-run vs standard) ----------
    rep = []
    for name, df in (('ICNALE', ic), ('Blog', bl), ('arXiv', ax)):
        for d in ['hello', 'fakespot']:
            pub = df[f'{d}_pub_flag']
            if f'{d}_orig_flag' in df:
                o = df[f'{d}_orig_flag']
                both = pub.notna() & o.notna()
                rep.append(dict(corpus=name, detector=NAMES[d], published_scored=int(pub.notna().sum()),
                                original_wrapper_scored=int(o.notna().sum()),
                                original_wrapper_failed=int((df[f'{d}_orig_error'] != '').sum()),
                                agree_pub_vs_orig=int((pub[both] == o[both]).sum()), compared=int(both.sum()),
                                agree_pub_vs_standard=int((pub[pub.notna()] == df.loc[pub.notna(), f'{d}_flag']).sum()),
                                published_unscored_over_512_tokens=int(((pub.isna()) & (df[f'{d}_n_tokens'] >= 512)).sum()),
                                published_unscored=int(pub.isna().sum())))
    pd.DataFrame(rep).to_csv(os.path.join(TAB, 'replication_open_source.csv'), index=False)

    # ---------------------------------- coverage ----------------------------------
    cov = []
    for name, df in (('ICNALE', ic), ('Blog', bl), ('arXiv', ax)):
        for d in ALL8:
            c = cell(df[f'{d}_flag'])
            cov.append(dict(corpus=name, detector=NAMES[d], corpus_n=len(df), scored=c['n'], flagged=c['k'],
                            fpr=round(c['rate'], 2), lo=round(c['lo'], 2), hi=round(c['hi'], 2)))
    pd.DataFrame(cov).to_csv(os.path.join(TAB, 'coverage.csv'), index=False)

    # ------------------------------ ICNALE: country ------------------------------
    order = ['India', 'Nepal', 'Bangladesh', 'Brunei', 'Vietnam', 'Cambodia', 'Myanmar', 'Malaysia', 'Laos',
             'Singapore', 'Indonesia', 'Philippines', 'Thailand', 'Korea', 'Japan', 'Hong Kong', 'China', 'Taiwan',
             'Pakistan', 'UAE', 'Native English']
    t = by_group(ic, 'Country', ALL8, order)
    t.to_csv(os.path.join(TAB, 'icnale_country.csv'), index=False)
    res['icnale_country_range'] = ranges(t, ALL8)
    res['icnale_country_range_L2only'] = ranges(t, ALL8, exclude=['Native English'])
    res['icnale_overall'] = {d: cell(ic[f'{d}_flag']) for d in ALL8}

    # ------------------------------ ICNALE: region ------------------------------
    ic['RegionLab'] = ic.Region
    t = by_group(ic, 'RegionLab', ALL8, ['South Asia', 'Southeast Asia', 'East Asia', 'Middle East', 'Native English'])
    t.to_csv(os.path.join(TAB, 'icnale_region.csv'), index=False)
    res['icnale_region_range'] = ranges(t, ALL8)

    # ------------------------------ ICNALE: CEFR ------------------------------
    t = by_group(ic, 'Proficiency', ALL8, ['A2', 'B1_1', 'B1_2', 'B2+', 'Native'])
    t.to_csv(os.path.join(TAB, 'icnale_cefr.csv'), index=False)
    res['icnale_cefr_b2_minus_a2'] = {d: float(t.loc[t.Proficiency == 'B2+', f'{d}_rate'].iloc[0]
                                               - t.loc[t.Proficiency == 'A2', f'{d}_rate'].iloc[0]) for d in ALL8}
    res['icnale_cefr_range_L2'] = ranges(t[t.Proficiency != 'Native'], ALL8)

    # ------------------------- ICNALE: country x CEFR cells -------------------------
    rows = []
    for (c, p), sub in ic[ic.Proficiency != 'Native'].groupby(['Country', 'Proficiency']):
        for d in MAIN:
            x = cell(sub[f'{d}_flag'])
            if x['n'] == 0:
                continue
            rows.append(dict(Country=c, CEFR=p, cell_size=len(sub), detector=NAMES[d], scored=x['n'], flagged=x['k'],
                             rate=round(x['rate'], 1), lo=round(x['lo'], 1), hi=round(x['hi'], 1),
                             reportable=x['n'] >= 30))
    cells = pd.DataFrame(rows)
    cells.to_csv(os.path.join(TAB, 'icnale_country_cefr_cells.csv'), index=False)

    # ------------------------------ Blog ------------------------------
    bl['AgeLab'] = bl['Age Group']
    bl['GA'] = bl.Gender.str.capitalize() + ', ' + bl['Age Group']
    tg = by_group(bl, 'Gender', ALL8, ['male', 'female'])
    ta = by_group(bl, 'AgeLab', ALL8, ['13-17', '23-27', '33-47', '48+'])
    tga = by_group(bl, 'GA', ALL8, [f'{g}, {a}' for g in ('Male', 'Female') for a in ('13-17', '23-27', '33-47', '48+')])
    for nm, tt in (('blog_gender', tg), ('blog_age', ta), ('blog_gender_age', tga)):
        tt.to_csv(os.path.join(TAB, f'{nm}.csv'), index=False)
    res['blog_overall'] = {d: cell(bl[f'{d}_flag']) for d in ALL8}
    # Ranges over the six gender x age cells with n >= 30 per detector, and over all eight cells.
    big = tga[~tga.GA.str.contains('48')]
    res['blog_range_gender_age_excl48'] = ranges(big, ALL8)
    res['blog_range_gender_age_all'] = ranges(tga, ALL8)
    res['blog_range_gender'] = ranges(tg, ALL8)
    res['blog_range_age_excl48'] = ranges(ta[ta.AgeLab != '48+'], ALL8)

    # ------------------------------ arXiv ------------------------------
    t = by_group(ax, 'Academic_Field', ALL8)
    t.to_csv(os.path.join(TAB, 'arxiv_field.csv'), index=False)
    res['arxiv_field_range'] = ranges(t, ALL8)
    res['arxiv_overall'] = {d: cell(ax[f'{d}_flag']) for d in ALL8}
    groups = {'Formal Sciences': ['Mathematics', 'Computer Science', 'Statistics'],
              'Physical Sciences': ['High Energy Physics', 'Condensed Matter', 'Astrophysics'],
              'Social Sciences': ['Quantitative Finance', 'Economics'], 'Life Sciences': ['Quantitative Biology'],
              'Applied Sciences': ['Electrical Engineering'], 'Interdisciplinary': ['Other']}
    ax['FieldGroup'] = ax.Academic_Field.map({f: g for g, fs in groups.items() for f in fs})
    by_group(ax, 'FieldGroup', ALL8, list(groups)).to_csv(os.path.join(TAB, 'arxiv_fieldgroup.csv'), index=False)

    # ------------------------------ DAIGT / DACTYL ------------------------------
    if not os.path.exists(os.path.join(R, 'runs', 'daigt_standard.csv')):
        print('DAIGT outputs not yet available; skipping DAIGT/DACTYL')
        return finish(res, ic)
    ev = pd.read_csv(os.path.join(R, '..', 'DAIGT', 'eval_dataset_1000.csv'))
    run = pd.read_csv(os.path.join(R, 'runs', 'daigt_standard.csv')).set_index('text_id')
    ev = ev.set_index('id')
    pub = {'originality': 'originality_pred', 'gptzero': 'gptzero_pred', 'zerogpt': 'zerogpt_pred',
           'hello': 'hello_simpleai_pred', 'fakespot': 'fakespot_pred'}
    for d, c in pub.items():
        ev[f'{d}_pub_flag'] = ev[c].map({'ChatGPT': 1.0, 'Human': 0.0})
    for d in ['originality', 'gptzero', 'zerogpt']:
        ev[f'{d}_flag'] = ev[f'{d}_pub_flag']
    ev['originality_p'] = ev.originality_score
    for d in LOCAL + ['e5']:
        ev[f'{d}_flag'] = run.loc[ev.index, f'{d}_pred_ai'].astype(float)
        ev[f'{d}_p_ai'] = run.loc[ev.index, f'{d}_p_ai']
    ev.drop(columns=['text']).to_csv(os.path.join(R, 'data', 'daigt_final_noText.csv'))
    ai, hu = ev[ev['class'] == 'AI'], ev[ev['class'] == 'Human']
    rows = []
    for g in ['mistral7binstruct_v1', 'mistral7binstruct_v2', 'chat_gpt_moth', 'llama2_chat', 'kingki19_palm', 'ALL']:
        sub = ai if g == 'ALL' else ai[ai.generator == g]
        r = {'generator': g}
        for d in ALL8:
            c = cell(sub[f'{d}_flag'])
            r.update({f'{d}_{k}': v for k, v in c.items()})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, 'daigt_tpr.csv'), index=False)
    res['daigt_tpr'] = {d: cell(ai[f'{d}_flag']) for d in ALL8}
    res['daigt_fpr'] = {d: cell(hu[f'{d}_flag']) for d in ALL8}
    res['daigt_hello_fakespot_agreement'] = {
        d: dict(ai=int((ai[f'{d}_pub_flag'] == ai[f'{d}_flag']).sum()), hu=int((hu[f'{d}_pub_flag'] == hu[f'{d}_flag']).sum()))
        for d in ['hello', 'fakespot']}
    # P(AI) bins on AI essays
    bins = np.linspace(0, 1, 11)
    pb = {}
    scorecol = {'originality': 'originality_score', 'gptzero': 'gptzero_score', 'zerogpt': 'zerogpt_score'}
    for d in ALL8:
        if d in scorecol:
            v = ai[scorecol[d]].where(ai[f'{d}_flag'].notna())
            if d == 'zerogpt' and v.max() > 1.5:
                v = v / 100
        else:
            v = ai[f'{d}_p_ai']
        v = v.dropna()
        h = np.histogram(v.clip(0, 1), bins=bins)[0]
        pb[d] = dict(n=int(len(v)), counts=h.tolist())
    res['daigt_pai_bins'] = pb

    if not os.path.exists(os.path.join(R, 'runs', 'dactyl_standard.csv')):
        return finish(res, ic)
    dac = pd.read_csv(os.path.join(R, 'data', 'dactyl_ai_sample.csv'))
    drun = pd.read_csv(os.path.join(R, 'runs', 'dactyl_standard.csv'))
    assert (dac.text_id.values == drun.text_id.values).all()
    for d in LOCAL + ['e5']:
        dac[f'{d}_flag'] = drun[f'{d}_pred_ai'].astype(float)
    rows = []
    for (dom, g), sub in dac.groupby(['domain', 'model']):
        r = {'domain': dom, 'generator': g}
        for d in LOCAL + ['e5']:
            c = cell(sub[f'{d}_flag'])
            r.update({f'{d}_{k}': v for k, v in c.items()})
        rows.append(r)
    for dom, sub in dac.groupby('domain'):
        r = {'domain': dom, 'generator': 'ALL'}
        for d in LOCAL + ['e5']:
            c = cell(sub[f'{d}_flag'])
            r.update({f'{d}_{k}': v for k, v in c.items()})
        rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(TAB, 'dactyl_tpr.csv'), index=False)

    return finish(res, ic)


def finish(res, ic):
    # Indian B2+ cell
    ib = ic[(ic.Country == 'India') & (ic.Proficiency == 'B2+')]
    res['india_b2'] = {d: cell(ib[f'{d}_flag']) for d in ALL8}
    res['india_b2_cellsize'] = len(ib)
    # native-English ICNALE row (Table 16)
    nat = ic[ic.Country == 'Native English']
    res['icnale_native'] = {d: cell(nat[f'{d}_flag']) for d in ALL8}
    ind = ic[ic.Country == 'India']
    res['icnale_india'] = {d: cell(ind[f'{d}_flag']) for d in ALL8}

    with open(os.path.join(TAB, 'results_r5.json'), 'w') as f:
        json.dump(res, f, indent=1, default=float)
    print(json.dumps({k: res[k] for k in ['icnale_country_range', 'icnale_cefr_b2_minus_a2', 'blog_range_gender_age_excl48',
                                          'arxiv_field_range']}, indent=1, default=lambda x: round(float(x), 2)))
    return res


if __name__ == '__main__':
    main()
