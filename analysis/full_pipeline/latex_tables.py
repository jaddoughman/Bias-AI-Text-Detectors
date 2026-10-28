"""Write LaTeX table bodies for the R5 manuscript and supporting tables from tables/*.csv.

Each main-text table is written to revision03/manuscript/tables/<name>.tex and \\input in
manuscript_src.tex; the tables are new in R5 and are printed in red in the tracked copy.
"""
import json
import os

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
TAB = os.path.join(R, 'tables')
OUT = os.path.join(R, 'manuscript', 'tables')
os.makedirs(OUT, exist_ok=True)

MAIN = ['originality', 'zerogpt', 'gptzero', 'hello', 'fakespot', 'roberta_large', 'radar']
HEAD = {'originality': 'Orig.AI', 'zerogpt': 'ZeroGPT', 'gptzero': 'GPTZero', 'hello': 'Hello-S.',
        'fakespot': 'Fakespot', 'roberta_large': 'RoB-L', 'radar': 'RADAR', 'e5': 'e5-small'}
FULL = {'originality': 'Originality.AI', 'zerogpt': 'ZeroGPT', 'gptzero': 'GPTZero', 'hello': 'Hello-SimpleAI',
        'fakespot': 'Fakespot', 'roberta_large': 'RoBERTa-large', 'radar': 'RADAR', 'e5': 'e5-small-lora'}


def f1(x):
    return '--' if x is None or (isinstance(x, float) and np.isnan(x)) else f'{x:.1f}'


def num(n):
    return f'{n:,}'.replace(',', '{,}')


def header(dets, first='', extra_first=None):
    cols = ' & '.join(r'\textbf{' + HEAD[d] + '}' for d in dets)
    return r'\textbf{' + first + '} & ' + cols


def write(name, s):
    import re
    s = re.sub(r'(?<=[\s&({])-(?=\d)', r'$-$', s)   # typographic minus for negative numbers
    with open(os.path.join(OUT, name + '.tex'), 'w') as f:
        f.write(s)


def two_line(top, bottom):
    return r'\begin{tabular}[c]{@{}c@{}}' + top + r'\\[-2pt]{\scriptsize ' + bottom + r'}\end{tabular}'


def t_country():
    t = pd.read_csv(os.path.join(TAB, 'icnale_country.csv'))
    region = {'India': 'S Asia', 'Nepal': 'S Asia', 'Bangladesh': 'S Asia', 'Pakistan': 'S Asia', 'Brunei': 'SE Asia',
              'Vietnam': 'SE Asia', 'Cambodia': 'SE Asia', 'Myanmar': 'SE Asia', 'Malaysia': 'SE Asia', 'Laos': 'SE Asia',
              'Singapore': 'SE Asia', 'Indonesia': 'SE Asia', 'Philippines': 'SE Asia', 'Thailand': 'SE Asia',
              'Korea': 'E Asia', 'Japan': 'E Asia', 'Hong Kong': 'E Asia', 'China': 'E Asia', 'Taiwan': 'E Asia',
              'UAE': 'Middle East', 'Native English': 'Native'}
    lines = [r'\begin{tabular}{lrrrrrrrl}', r'\toprule',
             r' & \multicolumn{3}{c}{\textbf{Commercial}} & \multicolumn{4}{c}{\textbf{Open-source}} & \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-8}',
             header(MAIN, 'Country') + r' & \textbf{Region} \\', r'\midrule']
    for _, r in t.iterrows():
        lines.append(r['Country'] + ' & ' + ' & '.join(f1(r[f'{d}_rate']) for d in MAIN) + f" & {region[r['Country']]} \\\\")
    lines.append(r'\midrule')
    lines.append(r'\textbf{Range (pp)} & ' + ' & '.join(r'\textbf{' + f1(t[f'{d}_rate'].max() - t[f'{d}_rate'].min()) + '}' for d in MAIN) + r' & \\')
    lines.append(r'Scored ($n$) & ' + ' & '.join(num(int(t[f'{d}_n'].sum())) for d in MAIN) + r' & \\')
    lines.append(r'$n$ per country & ' + ' & '.join(f"{int(t[f'{d}_n'].min())}--{int(t[f'{d}_n'].max())}" for d in MAIN) + r' & \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_country', '\n'.join(lines))


def t_region():
    t = pd.read_csv(os.path.join(TAB, 'icnale_region.csv'))
    lines = [r'\begin{tabular}{lccccccc}', r'\toprule', header(MAIN, 'Region') + r' \\', r'\midrule']
    for _, r in t.iterrows():
        cells = []
        for d in MAIN:
            cells.append(two_line(f1(r[f'{d}_rate']), f"{f1(r[f'{d}_lo'])}--{f1(r[f'{d}_hi'])}; $n$={int(r[f'{d}_n'])}"))
        lines.append(r['RegionLab'] + ' & ' + ' & '.join(cells) + r' \\[2pt]')
    lines.append(r'\midrule')
    lines.append(r'\textbf{Range (pp)} & ' + ' & '.join(f1(t[f'{d}_rate'].max() - t[f'{d}_rate'].min()) for d in MAIN) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_region', '\n'.join(lines))


def t_cefr():
    t = pd.read_csv(os.path.join(TAB, 'icnale_cefr.csv'))
    lab = {'A2': 'A2 (Elementary)', 'B1_1': r'B1\_1 (Intermediate)', 'B1_2': r'B1\_2 (Intermediate)',
           'B2+': 'B2+ (Advanced)', 'Native': 'Native speaker'}
    lines = [r'\begin{tabular}{lccccccc}', r'\toprule', header(MAIN, 'CEFR level') + r' \\', r'\midrule']
    for _, r in t.iterrows():
        cells = [two_line(f1(r[f'{d}_rate']), f"{int(r[f'{d}_k'])}/{int(r[f'{d}_n'])}") for d in MAIN]
        lines.append(lab[r['Proficiency']] + ' & ' + ' & '.join(cells) + r' \\[2pt]')
    lines.append(r'\midrule')
    l2 = t[t.Proficiency != 'Native']
    a2 = t[t.Proficiency == 'A2'].iloc[0]
    b2 = t[t.Proficiency == 'B2+'].iloc[0]
    lines.append(r'\textbf{B2+ $-$ A2 (pp)} & ' + ' & '.join(r'\textbf{' + f'{b2[f"{d}_rate"] - a2[f"{d}_rate"]:+.1f}' + '}' for d in MAIN) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_cefr', '\n'.join(lines))


def t_india(res):
    lines = [r'\begin{tabular}{lrrcc}', r'\toprule',
             r'\textbf{Detector} & \textbf{Scored} & \textbf{Flagged} & \textbf{FPR (\%)} & \textbf{95\% Wilson CI} \\', r'\midrule']
    for d in MAIN:
        c = res['india_b2'][d]
        if c['n'] < 30:  # below the reporting threshold: S3 Table only
            continue
        lines.append(f"{FULL[d]} & {c['n']} & {c['k']} & {f1(c['rate'])} & {f1(c['lo'])}--{f1(c['hi'])} \\\\")
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_india', '\n'.join(lines))


def t_arxiv():
    t = pd.read_csv(os.path.join(TAB, 'arxiv_field.csv'))
    grp = {'Mathematics': 'Formal', 'Computer Science': 'Formal', 'Statistics': 'Formal', 'High Energy Physics': 'Physical',
           'Condensed Matter': 'Physical', 'Astrophysics': 'Physical', 'Quantitative Finance': 'Social', 'Economics': 'Social',
           'Quantitative Biology': 'Life', 'Electrical Engineering': 'Applied', 'Other': 'Interdisc.'}
    t = t.sort_values('fakespot_rate', ascending=False)
    lines = [r'\begin{tabular}{lrrrrrrrl}', r'\toprule', header(MAIN, 'Field') + r' & \textbf{Group} \\', r'\midrule']
    for _, r in t.iterrows():
        lines.append(r['Academic_Field'] + ' & ' + ' & '.join(f1(r[f'{d}_rate']) for d in MAIN) + f" & {grp[r['Academic_Field']]} \\\\")
    lines.append(r'\midrule')
    lines.append(r'\textbf{Range (pp)} & ' + ' & '.join(r'\textbf{' + f1(t[f'{d}_rate'].max() - t[f'{d}_rate'].min()) + '}' for d in MAIN) + r' & \\')
    lines.append(r'Scored ($n$) & ' + ' & '.join(str(int(t[f'{d}_n'].sum())) for d in MAIN) + r' & \\')
    lines.append(r'$n$ per field & ' + ' & '.join(f"{int(t[f'{d}_n'].min())}--{int(t[f'{d}_n'].max())}" for d in MAIN) + r' & \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_arxiv', '\n'.join(lines))


def t_blog(res):
    g = pd.read_csv(os.path.join(TAB, 'blog_gender.csv'))
    a = pd.read_csv(os.path.join(TAB, 'blog_age.csv'))
    lines = [r'\begin{tabular}{lccccccc}', r'\toprule', header(MAIN, 'Group') + r' \\', r'\midrule',
             r'\multicolumn{8}{l}{\textit{\textbf{Gender}}} \\']
    for _, r in g.iterrows():
        cells = [two_line(f1(r[f'{d}_rate']), f"{int(r[f'{d}_k'])}/{int(r[f'{d}_n'])}") for d in MAIN]
        lines.append(r['Gender'].capitalize() + ' & ' + ' & '.join(cells) + r' \\[2pt]')
    lines += [r'\midrule', r'\multicolumn{8}{l}{\textit{\textbf{Age group}}} \\']
    lab = {'13-17': 'Adolescent (13--17)', '23-27': 'Young adult (23--27)', '33-47': 'Adult (33--47)', '48+': r'48+$^{\ddagger}$'}
    for _, r in a.iterrows():
        cells = [two_line(f1(r[f'{d}_rate']), f"{int(r[f'{d}_k'])}/{int(r[f'{d}_n'])}") for d in MAIN]
        lines.append(lab[r['AgeLab']] + ' & ' + ' & '.join(cells) + r' \\[2pt]')
    lines.append(r'\midrule')
    o = res['blog_overall']
    lines.append(r'\textbf{Overall} & ' + ' & '.join(two_line(r'\textbf{' + f1(o[d]['rate']) + '}', f"{o[d]['k']}/{o[d]['n']}") for d in MAIN) + r' \\[2pt]')
    rg = res['blog_range_gender_age_excl48']
    lines.append(r'\textbf{Range, gender $\times$ age (pp)}$^{\S}$ & ' + ' & '.join(f1(rg[d]) for d in MAIN) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_blog', '\n'.join(lines))


def t_blog_ga():
    t = pd.read_csv(os.path.join(TAB, 'blog_gender_age.csv'))
    lines = [r'\begin{tabular}{llccccccc}', r'\toprule', r'\textbf{Gender} & ' + header(MAIN, 'Age group') + r' \\', r'\midrule']
    for i, (_, r) in enumerate(t.iterrows()):
        gnd, age = r['GA'].split(', ')
        cells = [two_line(f1(r[f'{d}_rate']), f"{int(r[f'{d}_k'])}/{int(r[f'{d}_n'])}") for d in MAIN]
        lines.append((gnd if i % 4 == 0 else '') + ' & ' + age.replace('-', '--') + ' & ' + ' & '.join(cells) + r' \\[2pt]')
        if i == 3:
            lines.append(r'\midrule')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_blog_ga', '\n'.join(lines))


def t_tpr(res):
    t = pd.read_csv(os.path.join(TAB, 'daigt_tpr.csv'))
    lab = {'mistral7binstruct_v1': r'\texttt{mistral7binstruct\_v1}', 'mistral7binstruct_v2': r'\texttt{mistral7binstruct\_v2}',
           'chat_gpt_moth': r'\texttt{chat\_gpt\_moth} (ChatGPT)', 'llama2_chat': r'\texttt{llama2\_chat}',
           'kingki19_palm': r'\texttt{kingki19\_palm} (PaLM)', 'ALL': r'\textbf{Overall}'}
    lines = [r'\begin{tabular}{lrrrrrrr}', r'\toprule', header(MAIN, 'Generator') + r' \\', r'\midrule']
    for _, r in t.iterrows():
        if r['generator'] == 'ALL':
            lines.append(r'\midrule')
        lines.append(lab[r['generator']] + ' & ' + ' & '.join(f1(r[f'{d}_rate']) for d in MAIN) + r' \\')
    lines.append(r'($n$) & ' + ' & '.join(f"({int(t[t.generator == 'ALL'][f'{d}_n'].iloc[0])})" for d in MAIN) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_tpr', '\n'.join(lines))


def t_pai(res):
    pb = res['daigt_pai_bins']
    lines = [r'\begin{tabular}{lrrrrrrr}', r'\toprule', header(MAIN, '$P(\\text{AI})$ bin') + r' \\',
             r' & ' + ' & '.join(f"($n$={pb[d]['n']})" for d in MAIN) + r' \\', r'\midrule']
    for i in range(10):
        lo, hi = i / 10, (i + 1) / 10
        cells = []
        for d in MAIN:
            c = pb[d]['counts'][i]
            cells.append(f"{100 * c / pb[d]['n']:.1f} ({c})")
        lines.append(f'{lo:.1f}--{hi:.1f} & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_pai', '\n'.join(lines))


def t_asym(res):
    lines = [r'\begin{tabular}{lcccc}', r'\toprule',
             r'\textbf{Detector} & \textbf{TPR} & \textbf{FPR} & \textbf{FPR} & \textbf{FPR} \\',
             r' & \textbf{(DAIGT AI essays)} & \textbf{(DAIGT humans)} & \textbf{(Native English, ICNALE)} & \textbf{(Indian L2, ICNALE)} \\',
             r'\midrule']

    def c(x):
        return two_line(f1(x['rate']), f"{x['k']}/{x['n']}; {f1(x['lo'])}--{f1(x['hi'])}")
    for d in MAIN:
        lines.append(FULL[d] + ' & ' + ' & '.join(c(res[k][d]) for k in ('daigt_tpr', 'daigt_fpr', 'icnale_native', 'icnale_india')) + r' \\[2pt]')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_asym', '\n'.join(lines))


def t_coverage():
    cov = pd.read_csv(os.path.join(TAB, 'coverage.csv'))
    order = ['Originality.AI', 'ZeroGPT', 'GPTZero', 'Hello-SimpleAI', 'Fakespot', 'RoBERTa-large (OpenAI)', 'RADAR', 'e5-small-lora']
    lines = [r'\begin{tabular}{lrrrrrrrrr}', r'\toprule',
             r'\textbf{Corpus} & \textbf{Texts} & ' + ' & '.join(r'\textbf{' + HEAD[d] + '}' for d in MAIN + ['e5']) + r' \\', r'\midrule']
    for c in ['ICNALE', 'Blog', 'arXiv']:
        sub = cov[cov.corpus == c].set_index('detector')
        n = int(sub.corpus_n.iloc[0])
        lines.append(f'{c} & {num(n)} & ' + ' & '.join(num(int(sub.loc[d, 'scored'])) for d in order) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    write('t_coverage', '\n'.join(lines))


def t_profiles():
    p = pd.read_csv(os.path.join(TAB, 'linguistic_profiles.csv'))
    reg = {'Bangladesh': 'S Asia', 'India': 'S Asia', 'Nepal': 'S Asia', 'Vietnam': 'SE Asia', 'Cambodia': 'SE Asia',
           'Myanmar': 'SE Asia', 'Brunei': 'SE Asia', 'Malaysia': 'SE Asia', 'Indonesia': 'SE Asia', 'Thailand': 'SE Asia', 'Japan': 'E Asia', 'China': 'E Asia'}
    lines = [r'\begin{tabular}{llrrrcccccc}', r'\toprule',
             r'\textbf{Country} & \textbf{Region} & \textbf{FPR (\%)} & $n_{\mathrm{c}}$ & $n_{\mathrm{m}}$ & \multicolumn{3}{c}{\textbf{Avg word length}} & \multicolumn{3}{c}{\textbf{FRE}} \\',
             r'\cmidrule(lr){6-8}\cmidrule(lr){9-11}',
             r' & & & & & Correct & Misclass. & $\Delta$ & Correct & Misclass. & $\Delta$ \\', r'\midrule']
    for det, lab in (('Originality.AI', '(A) Originality.AI'), ('ZeroGPT', '(B) ZeroGPT')):
        lines.append(r'\multicolumn{11}{l}{\textbf{' + lab + r'}} \\')
        sub = p[p.detector == det]
        for _, r in sub.iterrows():
            mis = r.n_misclassified > 0
            aw = f'{r.awl_mis:.2f}' if mis else '--'
            fr = f'{r.fre_mis:.1f}' if mis else '--'
            daw = f'{r.awl_mis - r.awl_correct:+.2f}' if mis else '--'
            dfr = f'{r.fre_mis - r.fre_correct:+.1f}' if mis else '--'
            lines.append(f'{r.country} & {reg[r.country]} & {r.fpr:.1f} & {r.n_correct} & {r.n_misclassified} & {r.awl_correct:.2f} & {aw} & {daw} & {r.fre_correct:.1f} & {fr} & {dfr} \\\\')
        lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'
    lines.append(r'\end{tabular}')
    write('t_profiles', '\n'.join(lines))


def main():
    res = json.load(open(os.path.join(TAB, 'results_r5.json')))
    t_profiles(); t_coverage(); t_country(); t_region(); t_cefr(); t_india(res); t_arxiv(); t_blog(res); t_blog_ga(); t_tpr(res); t_pai(res); t_asym(res)
    print('LaTeX tables written to', OUT)


if __name__ == '__main__':
    main()
