"""Export per-text detector outputs (S1 Dataset; github/outputs/).

ICNALE essays are identified by their corpus file name and their text is not exported
(ICNALE terms of use). Blog and arXiv texts are public and are exported with the outputs.
For each detector: <det>_flag (1 = AI, 0 = human, empty = not scored) and <det>_score
(P(AI) for locally run detectors; the vendor's score for commercial detectors, as returned
in September 2025: Originality.AI and GPTZero on [0, 1], ZeroGPT on [0, 100]).
"""
import os
import shutil
import zipfile

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, '..')
OUT = os.path.join(R, 'S1_Dataset')
GH = os.path.join(R, '..', 'github', 'outputs')
COMM = ['originality', 'zerogpt', 'gptzero']
LOCAL = ['hello', 'fakespot', 'roberta_large', 'radar', 'e5']
FEATS = ['Word Count', 'Sentence Count', 'Avg Word Length', 'Lexical Diversity', 'Avg Sentence Length',
         'Flesch Reading Ease', 'Flesch-Kincaid Grade Level', 'Gunning Fog Index', 'Paragraph Count',
         'Avg Sentences per Paragraph', 'Avg Words per Paragraph', 'Avg Paragraph Length']


def commercial_score(df, d):
    """Vendor score as stored by the 2025 wrapper: P(AI) for Originality.AI and GPTZero (both on [0, 1])
    and the AI percentage for ZeroGPT, regardless of the predicted label; unscored texts are empty."""
    return df[f'{d}_score'].where(df[f'{d}_label'].isin(['ChatGPT', 'Human']))


def build(ds, keep, key):
    m = pd.read_csv(os.path.join(R, 'data', f'{ds}_master.csv'))
    run = pd.read_csv(os.path.join(R, 'runs', f'{ds}_standard.csv'))
    assert (m.text_id.values == run.text_id.values).all()
    out = m[keep].copy()
    for d in COMM:
        out[f'{d}_flag'] = m[f'{d}_label'].map({'ChatGPT': 1, 'Human': 0}).astype('Int64')
        out[f'{d}_score'] = commercial_score(m, d)
    for d in LOCAL:
        out[f'{d}_flag'] = run[f'{d}_pred_ai'].astype(bool).astype(int)
        out[f'{d}_score'] = run[f'{d}_p_ai'].round(6)
    for d in ['hello', 'fakespot']:
        out[f'{d}_published_2025_flag'] = m[f'{d}_label'].map({'ChatGPT': 1, 'Human': 0}).astype('Int64')
    return out


def daigt():
    """DAIGT v2 validation sample (500 AI + 500 human essays), same column convention as the other files."""
    ev = pd.read_csv(os.path.join(R, 'data', 'daigt_final_noText.csv'))
    out = ev[['id', 'class', 'generator', 'source_corpus', 'prompt_name', 'char_len']].copy()
    for d, c in (('originality', 'originality_pred'), ('zerogpt', 'zerogpt_pred'), ('gptzero', 'gptzero_pred')):
        out[f'{d}_flag'] = ev[c].map({'ChatGPT': 1, 'Human': 0}).astype('Int64')
        out[f'{d}_score'] = ev[f'{d}_score'].where(ev[c].isin(['ChatGPT', 'Human']))
    for d in LOCAL:
        out[f'{d}_flag'] = ev[f'{d}_flag'].astype(int)
        out[f'{d}_score'] = ev[f'{d}_p_ai'].round(6)
    for d, c in (('hello', 'hello_simpleai_pred'), ('fakespot', 'fakespot_pred')):
        out[f'{d}_published_2025_flag'] = ev[c].map({'ChatGPT': 1, 'Human': 0}).astype('Int64')
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(GH, exist_ok=True)
    ic = build('icnale', ['text_id', 'filename', 'Country', 'Proficiency', 'prompt', 'gptzero_in_sample'] + FEATS, 'filename')
    ic = ic.rename(columns={'filename': 'icnale_file'})
    ic['Country'] = ic.Country.replace({'English Native Speakers': 'Native English'})
    files = {'icnale_outputs.csv': ic,
             'blog_outputs.csv': build('blog', ['text_id', 'Gender', 'Age', 'Age Group', 'Text'] + FEATS, 'text_id'),
             'arxiv_outputs.csv': build('arxiv', ['text_id', 'ArXiv_ID', 'Academic_Field', 'Text'] + FEATS, 'ArXiv_ID')}
    files['daigt_outputs.csv'] = daigt()
    dac = pd.read_csv(os.path.join(R, 'data', 'dactyl_ai_sample.csv')).drop(columns=['Text'])
    drun = pd.read_csv(os.path.join(R, 'runs', 'dactyl_standard.csv'))
    for d in LOCAL:
        dac[f'{d}_flag'] = drun[f'{d}_pred_ai'].astype(bool).astype(int)
        dac[f'{d}_score'] = drun[f'{d}_p_ai'].round(6)
    files['dactyl_outputs.csv'] = dac
    wsf = os.path.join(R, 'runs', 'icnale_ws_standard.csv')
    if os.path.exists(wsf):
        ws = pd.read_csv(wsf)
        w = ic[['text_id', 'icnale_file', 'Country']].copy()
        for d in ['hello', 'fakespot', 'roberta_large', 'radar']:
            w[f'{d}_flag_whitespace_normalised'] = ws[f'{d}_pred_ai'].astype(bool).astype(int)
            w[f'{d}_score_whitespace_normalised'] = ws[f'{d}_p_ai'].round(6)
        files['icnale_whitespace_outputs.csv'] = w
    for name, df in files.items():
        df.to_csv(os.path.join(OUT, name), index=False)
        shutil.copy(os.path.join(OUT, name), os.path.join(GH, name))
    readme = open(os.path.join(R, '..', 'github', 'README.md')).read()
    sec = readme[readme.index('## Per-text outputs'):readme.index('## Reproducing the R5 analysis')]
    open(os.path.join(OUT, 'README.md'), 'w').write('# S1 Dataset: per-text detector outputs\n\n' + sec)
    with zipfile.ZipFile(os.path.join(R, 'supporting', 'S1_Dataset.zip'), 'w', zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(OUT)):
            z.write(os.path.join(OUT, f), arcname=f'S1_Dataset/{f}')
    print({k: len(v) for k, v in files.items()})


if __name__ == '__main__':
    main()
