"""Generate the supporting tables (standalone LaTeX, compiled to PDF) for revision R5.

S1  Regional feature deltas, with group sizes (rebuilt from per-text data)
S2  Distribution separation for higher-FPR countries (Originality.AI), with group sizes
S3  All ICNALE country x CEFR cells: flagged/scored per detector (cells with n < 30 marked)
S7  Per-detector coverage of every corpus, reasons, and replication of published outputs
S8  GPTZero outputs excluded at collection: per-country sensitivity
S9  Cohen's d, all 12 features x 7 detectors, with group sizes
S10 Conditional analyses (achievement-penalty decomposition, region gap, matched comparison,
    intersection check, line-break normalisation)
S11 Detection rates on contemporary (2024-2025) generators (DACTYL)
S12 Prompt-effect check
"""
import json
import os
import subprocess

import numpy as np
import pandas as pd

from common import REGION, NAMES, wilson

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
TAB = os.path.join(R, 'tables')
OUT = os.path.join(R, 'supporting')
os.makedirs(OUT, exist_ok=True)
MAIN = ['originality', 'zerogpt', 'gptzero', 'hello', 'fakespot', 'roberta_large', 'radar']
SHORT = {'originality': 'Orig.AI', 'zerogpt': 'ZeroGPT', 'gptzero': 'GPTZero', 'hello': 'Hello-S.',
         'fakespot': 'Fakespot', 'roberta_large': 'RoB-L', 'radar': 'RADAR', 'e5': 'e5'}

PRE = r"""\documentclass[10pt,letterpaper]{article}
\usepackage[margin=0.8in]{geometry}
\usepackage{booktabs,longtable,array,caption}
\captionsetup{labelformat=empty,labelsep=none}
\begin{document}
\sloppy
"""
POST = r"""
\end{document}
"""


def f1(x, nd=1):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.{nd}f}'


def esc(s):
    return str(s).replace('_', r'\_').replace('&', r'\&').replace('%', r'\%')


def build(name, body):
    path = os.path.join(OUT, name + '.tex')
    with open(path, 'w') as f:
        f.write(PRE + body + POST)
    for _ in range(2):
        subprocess.run(['pdflatex', '-interaction=nonstopmode', name + '.tex'], cwd=OUT,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path


def load():
    m = pd.read_csv(os.path.join(R, 'data', 'icnale_final_noText.csv'))
    return m


def s1(m):
    rows = []
    for d in ['originality', 'zerogpt', 'gptzero']:
        for reg in ['South Asia', 'Southeast Asia', 'East Asia']:
            t = m[(m.Region == reg) & m[f'{d}_flag'].notna()]
            a, b = t[t[f'{d}_flag'] == 0], t[t[f'{d}_flag'] == 1]
            dl = lambda f: (b[f].mean() - a[f].mean()) if len(b) >= 3 else np.nan
            rows.append((NAMES[d], reg, len(t), len(b), 100 * len(b) / len(t), dl('Avg Word Length'),
                         dl('Flesch Reading Ease'), dl('Lexical Diversity')))
    body = [r'\noindent\textbf{S1 Table. Regional linguistic feature differences between misclassified and correctly classified ICNALE essays, by commercial detector.}',
            r'\medskip', '',
            r'\noindent FPR is the percentage of the region\textquotesingle s scored human-authored essays flagged as AI-generated. $\Delta$ columns give the mean of misclassified minus correctly classified essays for average word length, Flesch Reading Ease (FRE) and lexical diversity (type--token ratio). Differences are not computed when fewer than three essays were misclassified (--). $n$ = essays scored; $n_{\mathrm{mis}}$ = essays misclassified. Recomputed for the current version from the per-text outputs (S1 Dataset); the previous version of this table omitted group sizes.',
            r'\bigskip', '', r'\begin{tabular}{llrrrrrr}', r'\toprule',
            r'\textbf{Detector} & \textbf{Region} & $n$ & $n_{\mathrm{mis}}$ & \textbf{FPR (\%)} & $\Delta$\textbf{Word len.} & $\Delta$\textbf{FRE} & $\Delta D_{lex}$ \\', r'\midrule']
    for r in rows:
        body.append(f'{r[0]} & {r[1]} & {r[2]} & {r[3]} & {f1(r[4])} & {f1(r[5], 2)} & {f1(r[6])} & {f1(r[7], 3)} \\\\')
    body += [r'\bottomrule', r'\end{tabular}']
    build('S1_Table', '\n'.join(body))


def s2(m):
    def si(a, b):
        iqr = (np.subtract(*np.percentile(a, [75, 25])) + np.subtract(*np.percentile(b, [75, 25]))) / 2
        return abs(np.median(a) - np.median(b)) / iqr if iqr > 0 else np.nan

    def overlap(a, b):
        qa, qb = np.percentile(a, [25, 75]), np.percentile(b, [25, 75])
        return 'Yes' if (qa[0] <= qb[1] and qb[0] <= qa[1]) else 'No'
    body = [r'\noindent\textbf{S2 Table. Distribution separation analysis for higher-FPR countries (Originality.AI and ZeroGPT).}',
            r'\medskip', '',
            r'\noindent Separation Index (SI) = $|$median(misclassified) $-$ median(correct)$|$ / pooled IQR, where the pooled IQR is the mean of the interquartile ranges of the two groups (the definition used in earlier versions); IQR overlap indicates whether the interquartile ranges of the two groups intersect. $n_{\mathrm{c}}$ / $n_{\mathrm{m}}$ = correctly classified / misclassified essays. With misclassified groups of about ten essays (Originality.AI), one or two atypical essays can move a median substantially; the ZeroGPT rows, with roughly three times as many scored essays, are given for comparison.',
            r'\bigskip', '', r'\begin{tabular}{lllrrcc}', r'\toprule',
            r'\textbf{Detector} & \textbf{Country} & \textbf{Feature} & $n_{\mathrm{c}}$ & $n_{\mathrm{m}}$ & \textbf{IQR overlap} & \textbf{SI} \\', r'\midrule']
    for d in ['originality', 'zerogpt']:
        for c in ['Bangladesh', 'India', 'Nepal', 'Vietnam']:
            t = m[(m.Country == c) & m[f'{d}_flag'].notna()]
            a, b = t[t[f'{d}_flag'] == 0], t[t[f'{d}_flag'] == 1]
            for f in ['Avg Word Length', 'Flesch Reading Ease', 'Lexical Diversity']:
                body.append(f'{NAMES[d]} & {c} & {f} & {len(a)} & {len(b)} & {overlap(a[f], b[f])} & {si(a[f].values, b[f].values):.2f} \\\\')
            body.append(r'\addlinespace[2pt]')
    body += [r'\bottomrule', r'\end{tabular}']
    build('S2_Table', '\n'.join(body))


def s3():
    c = pd.read_csv(os.path.join(TAB, 'icnale_country_cefr_cells.csv'))
    piv = {}
    for _, r in c.iterrows():
        piv.setdefault((r.Country, r.CEFR, r.cell_size), {})[r.detector] = (r.flagged, r.scored)
    dets = [NAMES[d] for d in MAIN]
    body = [r'\noindent\textbf{S3 Table. FPR in every ICNALE country~$\times$~CEFR cell, by detector (flagged/scored).}',
            r'\medskip', '',
            r'\noindent Entries are flagged/scored essays; cells where a detector scored fewer than 30 essays are shown in italics and are not used as evidence in the main text (Section~4.2.1 of the manuscript). Per-detector rates with 95\% Wilson intervals for every cell are given in S1 Dataset (file \texttt{icnale\_country\_cefr\_cells.csv}). ``Cell\textquotedblright{} is the number of sampled essays in the cell. The Indian B2+ cell discussed in the main text is in bold.',
            r'\bigskip', '', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}',
            r'\begin{longtable}{llr' + 'r' * len(dets) + '}', r'\toprule',
            r'\textbf{Country} & \textbf{CEFR} & \textbf{Cell} & ' + ' & '.join(r'\textbf{' + SHORT[d] + '}' for d in MAIN) + r' \\', r'\midrule', r'\endhead']
    for (cty, lev, size), v in sorted(piv.items(), key=lambda x: (x[0][0], x[0][1])):
        cells = []
        for dn in dets:
            if dn in v:
                k, n = v[dn]
                txt = f'{int(k)}/{int(n)}'
                cells.append(txt if n >= 30 else r'\textit{' + txt + '}')
            else:
                cells.append('--')
        line = f'{esc(cty)} & {esc(lev)} & {size} & ' + ' & '.join(cells) + r' \\'
        if cty == 'India' and lev == 'B2+':
            line = r'\textbf{India} & \textbf{B2+} & ' + f'{size} & ' + ' & '.join(cells) + r' \\'
        body.append(line)
    body += [r'\bottomrule', r'\end{longtable}']
    build('S3_Table', '\n'.join(body))


def s7():
    cov = pd.read_csv(os.path.join(TAB, 'coverage.csv'))
    rep = pd.read_csv(os.path.join(TAB, 'replication_open_source.csv'))
    reasons = json.load(open(os.path.join(TAB, 'coverage_reasons.json')))
    body = [r'\noindent\textbf{S7 Table. Per-detector coverage of the three human-authored corpora and replication of the published open-source outputs.}',
            r'\medskip', '', r'\noindent\textbf{(A) Texts scored, texts flagged, and overall FPR (95\% Wilson CI).} Commercial detectors: September 2025 outputs. Locally run detectors: re-run in October 2026 on every text with 512-token truncation at the revisions in S5 Table. e5-small-lora is excluded from the analysis under the 60\% screening rule and is shown for completeness.',
            r'\bigskip', '', r'\small', r'\begin{longtable}{llrrrl}', r'\toprule',
            r'\textbf{Corpus} & \textbf{Detector} & \textbf{Scored} & \textbf{Flagged} & \textbf{FPR \% (CI)} & \textbf{Reason for incomplete coverage} \\', r'\midrule', r'\endhead']
    for _, r in cov.iterrows():
        why = reasons.get(f"{r.corpus}|{r.detector}", '')
        body.append(f"{r.corpus} & {r.detector} & {r.scored:,}/{r.corpus_n:,} & {r.flagged} & {r.fpr:.2f} ({r.lo:.2f}--{r.hi:.2f}) & " +
                    r'\parbox[t]{5.2cm}{\raggedright ' + why + r'} \\')
    body += [r'\bottomrule', r'\end{longtable}', r'\normalsize', '',
             r'\bigskip', r'\noindent\textbf{(B) Replication of the published Hello-SimpleAI and Fakespot outputs.} The study wrapper used in 2025 truncated each text at 514 tokens, decoded it, and passed the decoded string to the model without truncation; texts of more than 512 tokens therefore raised an error and were skipped. We re-implemented that wrapper exactly (``original wrapper\textquotedblright{}) and compared its labels with the published labels, then re-scored every text with standard 512-token truncation (used for all results in R5).',
             r'\bigskip', '', r'\footnotesize', r'\begin{tabular}{llrrrrr}', r'\toprule',
             r'& & \textbf{Published} & \textbf{Original wrapper} & \textbf{Published vs} & \textbf{Published vs} & \textbf{Published unscored:} \\',
             r'\textbf{Corpus} & \textbf{Detector} & \textbf{scored} & \textbf{scored} & \textbf{original wrapper} & \textbf{2026 re-run} & \textbf{$\geq$512 tokens} \\', r'\midrule']
    for _, r in rep.iterrows():
        body.append(f"{r.corpus} & {r.detector} & {r.published_scored:,} & {r.original_wrapper_scored:,} & {r.agree_pub_vs_orig:,}/{r.compared:,} & {r.agree_pub_vs_standard:,}/{r.published_scored:,} & {r.published_unscored_over_512_tokens}/{r.published_unscored} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    build('S7_Table', '\n'.join(body))


def s8():
    t = pd.read_csv(os.path.join(TAB, 'gptzero_dropped.csv'))
    body = [r'\noindent\textbf{S8 Table. GPTZero outputs excluded at collection: sensitivity of the ICNALE results.}', r'\medskip', '',
            r'\noindent GPTZero was queried on the first 50 sampled essays per country (1{,}050 essays). The collection script retained only essays classified ``human\textquotedblright{} or ``AI\textquotedblright{} and discarded the remainder, which in GPTZero\textquotesingle s API are documents classified as ``mixed\textquotedblright{} (or, possibly, failed requests; the two cannot be distinguished after the fact). Twenty-four essays were discarded. All published GPTZero rates exclude them. The right-hand columns give the upper bound obtained by counting every discarded essay as flagged. The discarded essays were flagged by the other detectors at much higher rates than retained essays (bottom row), consistent with ``mixed\textquotedblright{} verdicts, so the published GPTZero rates understate GPTZero\textquotesingle s FPR in the affected countries.',
            r'\bigskip', '', r'\begin{tabular}{lrrrrrr}', r'\toprule',
            r'\textbf{Country} & \textbf{Discarded} & \textbf{Scored} & \textbf{Flagged} & \textbf{FPR (\%)} & \textbf{Upper bound $k/n$} & \textbf{Upper-bound FPR (\%)} \\', r'\midrule']
    for _, r in t.iterrows():
        body.append(f"{esc(r.Country)} & {r.dropped} & {r.n_scored} & {r.k} & {r.rate_published:.1f} & {r.k_if_flag}/{r.n_all} & {r.rate_if_flag:.1f} \\\\")
    body += [r'\midrule',
             f"\\textbf{{Country range (pp)}} & & & & {t.rate_published.max() - t.rate_published.min():.1f} & & {t.rate_if_flag.max() - t.rate_if_flag.min():.1f} \\\\",
             r'\bottomrule', r'\end{tabular}', '', r'\bigskip',
             r'\noindent ' + open(os.path.join(TAB, 'gptzero_dropped_note.txt')).read()]
    build('S8_Table', '\n'.join(body))


def s9():
    t = pd.read_csv(os.path.join(TAB, 'cohens_d.csv'))
    feats = ['Avg Word Length', 'Flesch Reading Ease', 'Lexical Diversity', 'Paragraph Count', 'Word Count', 'Sentence Count',
             'Avg Sentence Length', 'Flesch-Kincaid Grade Level', 'Gunning Fog Index', 'Avg Sentences per Paragraph',
             'Avg Words per Paragraph', 'Avg Paragraph Length']
    body = [r'\noindent\textbf{S9 Table. Cohen\textquotesingle s $d$ (misclassified minus correctly classified) for all 12 linguistic features, ICNALE, by detector.}',
            r'\medskip', '',
            r'\noindent Computed on all ICNALE essays each detector scored (pooled across countries). Positive values: the feature is larger in misclassified essays. Because pooled comparisons mix countries, these effect sizes partly reflect between-country differences; the within-country analyses are in S10 Table. Paragraph count is affected by corpus formatting: line breaks are present only in essays from eight country groups (Section~4.4.1 of the manuscript).',
            r'\bigskip', '', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{l' + 'r' * len(t) + '}', r'\toprule',
            r'\textbf{Feature} & ' + ' & '.join(r'\textbf{' + SHORT[k] + '}' for k in MAIN) + r' \\',
            r'$n_{\mathrm{correct}}$ & ' + ' & '.join(str(x) for x in t.n_correct) + r' \\',
            r'$n_{\mathrm{misclassified}}$ & ' + ' & '.join(str(x) for x in t.n_misclassified) + r' \\', r'\midrule']
    for f in feats:
        body.append(f + ' & ' + ' & '.join(f1(x, 2) for x in t[f]) + r' \\')
    body += [r'\bottomrule', r'\end{tabular}']
    build('S9_Table', '\n'.join(body))


def s10():
    c = pd.read_csv(os.path.join(TAB, 'conditional.csv'))
    mm = pd.read_csv(os.path.join(TAB, 'conditional_matched.csv'))
    ic = pd.read_csv(os.path.join(TAB, 'intersection_check.csv'))
    body = [r'\noindent\textbf{S10 Table. Conditional analyses of the proficiency and country disparities (ICNALE, L2 essays).}', r'\medskip', '',
            r'\noindent\textbf{(A) Achievement penalty and region gap, with and without conditioning on text features.} Linear probability models of the flag indicator (pp; 95\% bootstrap CIs, 1{,}000 resamples stratified by country). ``Marginal\textquotedblright{} is the raw B2+ minus A2 FPR difference on the same essays. ``Within-country\textquotedblright{} adds country fixed effects; ``+ features\textquotedblright{} adds standardised average word length, Flesch Reading Ease, lexical diversity, paragraph count and word count. The region gap is South Asia minus East Asia.',
            r'\bigskip', '', r'\scriptsize', r'\setlength{\tabcolsep}{3pt}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
            r'\textbf{Detector} & $n$ & \textbf{Marginal B2+$-$A2} & \textbf{Within-country} & \textbf{+ features} & \textbf{SA$-$EA gap} & \textbf{SA$-$EA + features} \\', r'\midrule']
    for _, r in c.iterrows():
        body.append(f"{r.detector} & {r.n} & {r.marginal_b2_a2:.1f} & {r.within_country_b2_a2:.1f} ({r.lo0:.1f}, {r.hi0:.1f}) & {r.within_country_b2_a2_features:.1f} ({r.lo1:.1f}, {r.hi1:.1f}) & {r.sa_ea_gap:.1f} ({r.sa_lo0:.1f}, {r.sa_hi0:.1f}) & {r.sa_ea_gap_features:.1f} ({r.sa_lo1:.1f}, {r.sa_hi1:.1f}) \\\\")
    body += [r'\bottomrule', r'\end{tabular}', '', r'\bigskip', r'\normalsize',
             r'\noindent\textbf{(B) FPR inside and outside the high-risk feature region} (average word length $\geq$ 4.7 characters and Flesch Reading Ease $\leq$ 55). ``Share\textquotedblright{} is the percentage of the region\textquotesingle s essays that fall in the high-risk region.',
             r'\bigskip', '', r'\scriptsize', r'\begin{tabular}{llrrrcr}', r'\toprule',
             r'\textbf{Detector} & \textbf{Region} & \textbf{Share (\%)} & $n$ \textbf{in region} & \textbf{FPR in region (\%)} & \textbf{95\% CI} & \textbf{FPR outside (\%)} \\', r'\midrule']
    for _, r in mm[~mm.region.str.contains('matched')].iterrows():
        body.append(f"{r.detector} & {r.region} & {r.share_in_high_risk:.1f} & {int(r.n_high_risk)} & {f1(r.fpr)} & {f1(r.lo)}--{f1(r.hi)} & {f1(r.fpr_outside)} \\\\")
    body += [r'\bottomrule', r'\end{tabular}', '', r'\bigskip', r'\normalsize',
             r'\noindent\textbf{(C) Indian B2+ cell: observed FPR versus the FPR implied by additive country and CEFR effects} (penalised logistic regression, flag $\sim$ country + CEFR; one-sided binomial test of the observed count against the additive prediction).',
             r'\bigskip', '', r'\scriptsize', r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r'\textbf{Detector} & $n$ & \textbf{Flagged} & \textbf{Observed (\%)} & \textbf{Additive prediction (\%)} & \textbf{Excess (pp)} & $p$ \\', r'\midrule']
    for _, r in ic.iterrows():
        body.append(f"{r.detector} & {r.n} & {r.flagged} & {r.observed:.1f} & {r.predicted_additive:.1f} & {r.excess_pp:+.1f} & {r.p_one_sided:.3f} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    wsf = os.path.join(TAB, 'whitespace.csv')
    if os.path.exists(wsf):
        w = pd.read_csv(wsf)
        body += ['', r'\bigskip', r'\normalsize',
                 r'\noindent\textbf{(D) Line-break normalisation (locally run detectors).} Each ICNALE essay was re-scored after replacing every run of whitespace, including line breaks, with a single space. Line breaks occur only in essays from eight country groups (Bangladesh, Cambodia, India, Laos, Malaysia, Myanmar, Nepal, Vietnam).',
                 r'\bigskip', '', r'\scriptsize', r'\begin{tabular}{llrrrrr}', r'\toprule',
                 r'\textbf{Detector} & \textbf{Essays} & $n$ & \textbf{FPR original (\%)} & \textbf{FPR normalised (\%)} & \textbf{AI$\to$human} & \textbf{human$\to$AI} \\', r'\midrule']
        for _, r in w.iterrows():
            fl = '' if np.isnan(r.flips_to_human) else f'{int(r.flips_to_human)}'
            fa = '' if np.isnan(r.flips_to_ai) else f'{int(r.flips_to_ai)}'
            body.append(f"{r.detector} & {r.group} & {r.n} & {r.fpr_original:.1f} & {r.fpr_whitespace_normalised:.1f} & {fl} & {fa} \\\\")
        body += [r'\bottomrule', r'\end{tabular}']
    build('S10_Table', '\n'.join(body))


def s11():
    t = pd.read_csv(os.path.join(TAB, 'dactyl_tpr.csv'))
    d4 = ['hello', 'fakespot', 'roberta_large', 'radar', 'e5']
    body = [r'\noindent\textbf{S11 Table. Detection rates (TPR, \%) of the locally run detectors on text from eleven 2024--2025 generators (DACTYL test split).}', r'\medskip', '',
            r'\noindent AI-generated student essays (50 per generator) and scientific abstracts (30 per generator) sampled at random (seed 42, $\geq$500 characters) from the DACTYL test split \cite{thorat2025dactyl}; many of the essay prompts ask the model to imitate a particular student\textquotesingle s style. Commercial detectors could not be re-queried at their September 2025 versions and are not included. e5-small-lora is shown for completeness; its high TPR is uninformative given its 74.6\% FPR on human text.'.replace(r' \cite{thorat2025dactyl}', ' (Thorat and Caines, 2025, arXiv:2508.00619)'),
            r'\bigskip', '', r'\small', r'\begin{tabular}{ll' + 'r' * len(d4) + '}', r'\toprule',
            r'\textbf{Domain} & \textbf{Generator} & ' + ' & '.join(r'\textbf{' + SHORT[d] + '}' for d in d4) + r' \\', r'\midrule']
    for dom in ['student_essays', 'abstracts']:
        sub = t[t.domain == dom]
        sub = pd.concat([sub[sub.generator != 'ALL'], sub[sub.generator == 'ALL']])
        for _, r in sub.iterrows():
            g = r'\textbf{All generators}' if r.generator == 'ALL' else esc(r.generator)
            body.append(f"{'Essays' if dom == 'student_essays' else 'Abstracts'} & {g} & " + ' & '.join(f1(r[f'{d}_rate']) for d in d4) + r' \\')
        body.append(r'\midrule')
    body[-1] = r'\bottomrule'
    body += [r'\end{tabular}']
    build('S11_Table', '\n'.join(body))


def s12():
    t = pd.read_csv(os.path.join(TAB, 'prompt_check.csv'))
    body = [r'\noindent\textbf{S12 Table. Prompt effects versus country effects (ICNALE).}', r'\medskip', '',
            r'\noindent ICNALE essays respond to one of two prompts (PTJ: part-time jobs for college students; SMK: smoking ban in restaurants). For each detector: the country FPR range computed separately within each prompt; FPR by prompt; the share of variance in the flag indicator explained by country and by prompt ($R^2$ of linear probability models); and the prompt difference conditional on country. This replaces a summary statistic reported in Section~5.2 of the previous version of the manuscript, which we could not reproduce from the per-text outputs.',
            r'\bigskip', '', r'\small', r'\begin{tabular}{lrrrrrrr}', r'\toprule',
            r'\textbf{Detector} & \textbf{Range PTJ} & \textbf{Range SMK} & \textbf{FPR PTJ} & \textbf{FPR SMK} & $R^2$ \textbf{country} & $R^2$ \textbf{prompt} & \textbf{SMK$-$PTJ $|$ country} \\', r'\midrule']
    for _, r in t.iterrows():
        body.append(f"{r.detector} & {r.range_ptj:.1f} & {r.range_smk:.1f} & {r.fpr_ptj:.1f} & {r.fpr_smk:.1f} & {r.r2_country:.3f} & {r.r2_prompt:.3f} & {r.prompt_coef:+.1f} \\\\")
    body += [r'\bottomrule', r'\end{tabular}']
    build('S12_Table', '\n'.join(body))



def s6():
    """S6: the excluded e5-small-lora on all corpora (replaces the R4 sensitivity table)."""
    ic = pd.read_csv(os.path.join(TAB, 'icnale_country.csv'))
    ce = pd.read_csv(os.path.join(TAB, 'icnale_cefr.csv'))
    cov = pd.read_csv(os.path.join(TAB, 'coverage.csv'))
    tpr = pd.read_csv(os.path.join(TAB, 'daigt_tpr.csv'))
    bg = pd.read_csv(os.path.join(TAB, 'blog_gender.csv'))
    ax = pd.read_csv(os.path.join(TAB, 'arxiv_field.csv'))
    e = cov[cov.detector == 'e5-small-lora'].set_index('corpus')
    body = [r'\noindent\textbf{S6 Table. Results for the excluded system \texttt{MayZhou/e5-small-lora-ai-generated-detector}.}', r'\medskip', '',
            r'\noindent \texttt{e5-small-lora} is the only evaluated system excluded under the 60\% screening rule (Section~3.2.2 of the manuscript). Its convention (``Label\_1: AI-generated\textquotedblright{}) is documented on its model card and was applied correctly. Results are from the October 2026 re-run (512-token truncation; revision \texttt{483fc496}). In the previous version this table reported the ICNALE sensitivity analysis for RoBERTa-large and RADAR; both are now analysed in the main text.',
            r'\bigskip', '', r'\begin{center}\textbf{(A) Overall FPR on the three human-authored corpora}\end{center}',
            r'\begin{center}\begin{tabular}{lrrr}', r'\toprule', r'\textbf{Corpus} & \textbf{Scored} & \textbf{Flagged} & \textbf{FPR \% (95\% CI)} \\', r'\midrule']
    for c in ['ICNALE', 'Blog', 'arXiv']:
        r = e.loc[c]
        body.append(f'{c} & {r.scored:,} & {r.flagged:,} & {r.fpr:.2f} ({r.lo:.2f}--{r.hi:.2f}) \\\\')
    body += [r'\bottomrule', r'\end{tabular}\end{center}', '', r'\begin{center}\textbf{(B) ICNALE FPR by country and CEFR level}\end{center}',
             r'\begin{center}\small\begin{tabular}{lrr@{\hspace{2em}}lrr}', r'\toprule',
             r'\textbf{Country} & $n$ & \textbf{FPR (\%)} & \textbf{CEFR} & $n$ & \textbf{FPR (\%)} \\', r'\midrule']
    cl = ce[['Proficiency', 'e5_n', 'e5_rate']].values.tolist()
    for i, (_, r) in enumerate(ic.iterrows()):
        right = f'{esc(cl[i][0])} & {int(cl[i][1])} & {cl[i][2]:.1f}' if i < len(cl) else ' & & '
        body.append(f'{esc(r.Country)} & {int(r.e5_n)} & {r.e5_rate:.1f} & {right} \\\\')
    body += [r'\midrule', f"Country range (pp) & & {ic.e5_rate.max() - ic.e5_rate.min():.1f} & & & \\\\", r'\bottomrule', r'\end{tabular}\end{center}', '',
             r'\begin{center}\textbf{(C) Blog Corpus (by gender) and arXiv (field range); detection rate on DAIGT v2 AI essays}\end{center}',
             r'\begin{center}\small\begin{tabular}{lr}', r'\toprule']
    for _, r in bg.iterrows():
        body.append(f'Blog, {r.Gender} & {r.e5_rate:.1f}\\% ({int(r.e5_k)}/{int(r.e5_n)}) \\\\')
    body.append(f'arXiv, field FPR range & {ax.e5_rate.min():.1f}--{ax.e5_rate.max():.1f}\\% \\\\')
    o = tpr[tpr.generator == 'ALL'].iloc[0]
    body.append(f'DAIGT v2 AI essays, TPR & {o.e5_rate:.1f}\\% ({int(o.e5_k)}/{int(o.e5_n)}) \\\\')
    body += [r'\bottomrule', r'\end{tabular}\end{center}', '',
             r'\noindent Its near-complete detection of AI-generated text is uninformative given that it also flags most human-authored text in every group.']
    build('S6_Table', '\n'.join(body))


if __name__ == '__main__':
    m = load()
    s1(m); s2(m); s3(); s6(); s7(); s8(); s9(); s10(); s11(); s12()
    print('supporting tables in', OUT)
