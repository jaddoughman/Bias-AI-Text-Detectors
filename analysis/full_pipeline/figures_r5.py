"""Regenerate Figs 1-4 (Originality.AI) with group sizes, and S1 Fig (ZeroGPT replication).

Box: median and IQR; whiskers: 1.5 x IQR; points: individual essays (jittered).
Every group label carries its n so the reader can judge the precision of each box.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import REGION  # noqa: E402

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
OUT = os.path.join(R, 'figures')
os.makedirs(OUT, exist_ok=True)
C_OK, C_MIS = '#3B82C4', '#D9473B'
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 9, 'axes.labelsize': 8, 'font.family': 'DejaVu Sans'})

FEATS = [('Avg Word Length', 'Average word length (characters)'), ('Flesch Reading Ease', 'Flesch Reading Ease'),
         ('Lexical Diversity', 'Lexical diversity (type-token ratio)'), ('Paragraph Count', 'Paragraph count')]


def boxpanel(ax, groups, feat, title, ylim=None):
    data = [g[1][feat].dropna().values for g in groups]
    pos = np.arange(1, len(groups) + 1)
    bp = ax.boxplot(data, positions=pos, widths=0.6, patch_artist=True, showfliers=False,
                    medianprops=dict(color='black', lw=1))
    rng = np.random.default_rng(0)
    for i, (g, d) in enumerate(zip(groups, data)):
        col = g[2]
        bp['boxes'][i].set(facecolor=col, alpha=0.35, edgecolor=col)
        ax.scatter(pos[i] + rng.uniform(-0.18, 0.18, len(d)), d, s=3, color=col, alpha=0.6, lw=0)
    ax.set_xticks(pos)
    ax.set_xticklabels([f'{g[0]}\n(n={len(d)})' for g, d in zip(groups, data)], rotation=60, ha='right', fontsize=6.5)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis='y', lw=0.3, alpha=0.5)


def groups_by(df, det, col, values, label_fn=lambda v: v, both=True):
    out = []
    s = df[df[f'{det}_flag'].notna()]
    for v in values:
        sub = s[s[col] == v]
        ok, mis = sub[sub[f'{det}_flag'] == 0], sub[sub[f'{det}_flag'] == 1]
        out.append((f'{label_fn(v)} correct', ok, C_OK))
        if both and len(mis):
            out.append((f'{label_fn(v)} misclass.', mis, C_MIS))
    return out


def save(fig, name):
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, name + '.eps'), format='eps')
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=200)
    fig.savefig(os.path.join(OUT, name + '.tif'), dpi=300, pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig)


def figure_set(m, det, label, prefix):
    m = m.copy()
    m['Region'] = m.Country.map(REGION).fillna('Native English')
    hb = ['Brunei', 'Vietnam', 'India', 'Bangladesh', 'Nepal', 'Cambodia', 'Myanmar', 'Malaysia']
    # Fig 1: high-bias countries, four features
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.2))
    g = groups_by(m, det, 'Country', hb)
    for ax, (f, t) in zip(axes.flat, FEATS):
        boxpanel(ax, g, f, t)
    fig.suptitle(f'Linguistic features of correctly classified and misclassified essays, higher-FPR countries ({label})', fontsize=9)
    save(fig, f'{prefix}1')
    # Fig 2: four high-bias vs four low-bias countries, AWL and FRE
    fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.9))
    g = groups_by(m, det, 'Country', ['Bangladesh', 'India', 'Nepal', 'Vietnam'])
    g += [(f'{c} (all correct)' if (m[(m.Country == c) & m[f'{det}_flag'].notna()][f'{det}_flag'].sum() == 0) else c,
           m[(m.Country == c) & (m[f'{det}_flag'] == 0)], '#4C9F70') for c in ['Indonesia', 'Thailand', 'Japan', 'China']]
    for ax, (f, t) in zip(axes, FEATS[:2]):
        boxpanel(ax, g, f, t, ylim=(3.3, 5.9) if f == 'Avg Word Length' else (0, 100))
    fig.suptitle(f'Higher-FPR versus zero-FPR countries ({label})', fontsize=9)
    save(fig, f'{prefix}2')
    # Fig 3: regions
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.2))
    g = groups_by(m, det, 'Region', ['South Asia', 'Southeast Asia', 'East Asia', 'Middle East', 'Native English'])
    for ax, (f, t) in zip(axes.flat, FEATS):
        boxpanel(ax, g, f, t, ylim=(0, 100) if f == 'Flesch Reading Ease' else None)
    fig.suptitle(f'Regional feature distributions ({label})', fontsize=9)
    save(fig, f'{prefix}3')
    # Fig 4: CEFR
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.2))
    g = groups_by(m, det, 'Proficiency', ['A2', 'B1_1', 'B1_2', 'B2+'], label_fn=lambda v: v.replace('_', '.'))
    for ax, (f, t) in zip(axes.flat, FEATS):
        boxpanel(ax, g, f, t, ylim=(0, 100) if f == 'Flesch Reading Ease' else None)
    fig.suptitle(f'Feature distributions by CEFR level ({label})', fontsize=9)
    save(fig, f'{prefix}4')


if __name__ == '__main__':
    src = os.path.join(R, 'data', 'icnale_master.csv')
    m = pd.read_csv(src)
    figure_set(m, 'originality', 'Originality.AI', 'Fig')
    figure_set(m, 'zerogpt', 'ZeroGPT', 'S1_Fig_ZeroGPT_')
    print('figures written to', OUT)
