"""Conditional analyses for Section 4.4.3 (Reading A vs Reading B).

For each detector, on the L2 ICNALE essays it scored:

1. Achievement-penalty decomposition. Linear probability models
       M0: flag ~ CEFR + country FE
       M1: flag ~ CEFR + country FE + text features
   The B2+ minus A2 coefficient is the within-country achievement penalty; the share of
   it removed by adding the features is the portion statistically accounted for by the
   text's linguistic profile (a product-of-coefficients style decomposition; LPM
   coefficients are comparable across nested models, unlike logit coefficients).
   95% CIs by nonparametric bootstrap over essays (B = 1000, stratified by country).

2. Region contrast conditional on features. Same models with region in place of
   CEFR: the South Asia minus East Asia gap before and after conditioning on features.

3. Feature-matched comparison. Essays are stratified on the joint quintiles of
   average word length and Flesch Reading Ease (25 strata, quintiles computed on all
   L2 essays); within each stratum the FPR is compared across regions and CEFR levels,
   and the strata are pooled with stratum weights from the comparison group
   (direct standardisation). If Reading B held, region/CEFR gaps would persist among
   texts with matched feature profiles.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

FEATURES = ['Avg Word Length', 'Flesch Reading Ease', 'Lexical Diversity', 'Paragraph Count', 'Word Count']
SAFE = {f: f.replace(' ', '_').replace('-', '_') for f in FEATURES}


def prep(df, det):
    s = df[(df[f'{det}_flag'].notna()) & (df.Proficiency != 'Native')].copy()
    s['y'] = s[f'{det}_flag'].astype(float)
    for f, g in SAFE.items():
        s[g] = (s[f] - s[f].mean()) / s[f].std()
    s['CEFR'] = pd.Categorical(s.Proficiency, ['A2', 'B1_1', 'B1_2', 'B2+'])
    return s


def _gap(fit, term):
    return fit.params.get(term, np.nan)


FEAT_TERMS = ' + '.join(SAFE.values())
T_B2 = 'C(CEFR)[T.B2+]'


def _design(s, group_col, base, extra_feats):
    """Design matrix: intercept, group dummies (reference = base), country FE (optional), features."""
    parts = [np.ones((len(s), 1))]
    names = ['const']
    for lev in [l for l in pd.unique(s[group_col]) if l != base]:
        parts.append((s[group_col].values == lev).astype(float)[:, None]); names.append(f'{group_col}={lev}')
    if group_col != 'Region':
        for c in sorted(pd.unique(s.Country))[1:]:
            parts.append((s.Country.values == c).astype(float)[:, None]); names.append(f'C={c}')
    if extra_feats:
        for g in SAFE.values():
            parts.append(s[g].values[:, None]); names.append(g)
    return np.hstack(parts), names


def _ols_coef(X, y, names, term):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return beta[names.index(term)] if term in names else np.nan


def penalty(s):
    s = s.copy()
    s['CEFRs'] = s.Proficiency.astype(str)
    X0, n0 = _design(s, 'CEFRs', 'A2', False)
    X1, n1 = _design(s, 'CEFRs', 'A2', True)
    return _ols_coef(X0, s.y.values, n0, 'CEFRs=B2+'), _ols_coef(X1, s.y.values, n1, 'CEFRs=B2+')


def region_gap(s):
    s = s[s.Region.isin(['South Asia', 'East Asia', 'Southeast Asia', 'Middle East'])]
    X0, n0 = _design(s, 'Region', 'East Asia', False)
    X1, n1 = _design(s, 'Region', 'East Asia', True)
    return _ols_coef(X0, s.y.values, n0, 'Region=South Asia'), _ols_coef(X1, s.y.values, n1, 'Region=South Asia')


def boot(s, fn, B=1000, seed=7):
    rng = np.random.default_rng(seed)
    idx_by_c = [g.index.values for _, g in s.groupby('Country')]
    out = []
    for _ in range(B):
        ix = np.concatenate([rng.choice(ix, size=len(ix), replace=True) for ix in idx_by_c])
        b = s.loc[ix].reset_index(drop=True)
        try:
            out.append(fn(b))
        except Exception:
            continue
    return np.array(out)


def strata(all_l2):
    q_w = all_l2['Avg Word Length'].quantile([.2, .4, .6, .8]).values
    q_f = all_l2['Flesch Reading Ease'].quantile([.2, .4, .6, .8]).values
    return q_w, q_f


def matched(s, q_w, q_f, group_col, a, b):
    """Directly standardised FPR of groups a and b over the feature strata of group a."""
    s = s.copy()
    s['sw'] = np.digitize(s['Avg Word Length'], q_w)
    s['sf'] = np.digitize(s['Flesch Reading Ease'], q_f)
    s['stratum'] = s.sw * 5 + s.sf
    A, Bg = s[s[group_col] == a], s[s[group_col] == b]
    common = sorted(set(A.stratum) & set(Bg.stratum))
    wa = A[A.stratum.isin(common)].stratum.value_counts()
    ra = A.groupby('stratum').y.mean()
    rb = Bg.groupby('stratum').y.mean()
    w = wa / wa.sum()
    std_a = float((w * ra.reindex(w.index)).sum())
    std_b = float((w * rb.reindex(w.index)).sum())
    cover = A.stratum.isin(common).mean()
    return dict(raw_a=A.y.mean(), raw_b=Bg.y.mean(), std_a=std_a, std_b=std_b,
                n_a=int(A.stratum.isin(common).sum()), n_b=int(Bg.stratum.isin(common).sum()),
                share_a_matched=cover, n_strata=len(common))


def interaction_check(s, country='India', level='B2+'):
    """Observed FPR in a country x CEFR cell versus the FPR predicted by an additive
    (main-effects) logistic model flag ~ country + CEFR fitted to all L2 essays the detector
    scored. A cell rate well above the main-effects prediction would indicate a compounding
    (non-additive) intersectional effect; a rate near it indicates that the cell's elevation
    is what its country and CEFR level jointly imply."""
    import statsmodels.formula.api as smf
    fit = smf.logit('y ~ C(Country) + C(CEFR)', data=s).fit(disp=0)
    cell = s[(s.Country == country) & (s.Proficiency == level)]
    pred = fit.predict(cell).mean()
    return dict(n=len(cell), observed=cell.y.mean(), predicted_additive=pred)
