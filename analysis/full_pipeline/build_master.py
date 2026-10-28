"""Build per-text master tables joining every published detector output.

Outputs (revision03/data/):
  icnale_master.csv  2,080 ICNALE essays in sampling order, with filename, prompt,
                     CEFR level, and label/score for the five retained detectors
                     (NaN = not scored), plus gptzero_in_sample (the first 50 essays
                     per country, i.e. the 50-per-country GPTZero sample).
  blog_master.csv    2,000 Blog Authorship Corpus posts (gender-stratified sample).
  arxiv_master.csv   all arXiv abstracts scored by at least one detector.
"""
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..'))
OUT = os.path.join(ROOT, 'revision03', 'data')
sys.path.insert(0, HERE)
import icnale_sample  # noqa: E402

FILES = {
    'icnale': {'originality': 'originality', 'gptzero': 'gptzero_1000', 'zerogpt': 'zero_gpt',
               'hello': 'hello_simple_ai', 'fakespot': 'fakespot'},
    'blogs': {'originality': 'originality', 'gptzero': 'gptzero_1000', 'zerogpt': 'zerogpt',
              'hello': 'hello_simple_ai', 'fakespot': 'fakespot'},
    'arxiv': {'originality': 'originality', 'gptzero': 'gptzero_1000', 'zerogpt': 'zero_gpt',
              'hello': 'hello_simple_ai', 'fakespot': 'fakespot'},
}
FEATS = ['Word Count', 'Sentence Count', 'Avg Word Length', 'Lexical Diversity', 'Avg Sentence Length',
         'Flesch Reading Ease', 'Flesch-Kincaid Grade Level', 'Gunning Fog Index', 'Paragraph Count',
         'Avg Sentences per Paragraph', 'Avg Words per Paragraph', 'Avg Paragraph Length']


def read(ds, det):
    return pd.read_csv(os.path.join(ROOT, 'results', ds, FILES[ds][det] + '.csv'))


def attach(master, ds, key):
    for det in FILES[ds]:
        d = read(ds, det)
        assert d[key].is_unique, (ds, det)
        d = d.set_index(key)
        m = master[key].map(d['Predicted Label'])
        master[f'{det}_label'] = m
        master[f'{det}_flag'] = m.map({'ChatGPT': 1, 'Human': 0})
        master[f'{det}_score'] = master[key].map(d['Prediction Score'])
    return master


def icnale():
    H = read('icnale', 'hello')
    full = icnale_sample.load('sorted')
    full = full[full.text.str.len() > 500]
    # Three ICNALE essays occur twice in the corpus (identical text filed under both prompts,
    # India x2 and UAE x1); map metadata from the first occurrence and flag them.
    dup_texts = set(full.text[full.text.duplicated()])
    by_text = full.drop_duplicates('text').set_index('text')
    # The 100-per-country sample = Hello-SimpleAI rows (in sampling order) + the essays
    # the open-source wrappers could not score; recover the latter from the other files.
    others = pd.concat([read('icnale', d) for d in ('zerogpt', 'originality', 'gptzero')])
    extra = others.loc[~others.Text.isin(set(H.Text)), ['Country', 'Proficiency', 'Text']].drop_duplicates('Text')
    print('ICNALE essays not scored by the open-source wrappers:', len(extra), extra.Country.tolist())
    # Within-country order follows the published (sampling-order) file; the unscored essays
    # are appended. GPTZero's 50-per-country sample is the first 50 drawn per country.
    order = []
    for c in H.Country.drop_duplicates():
        ht = H.loc[H.Country == c, 'Text'].tolist()
        ex = extra.loc[extra.Country == c, 'Text'].tolist()
        order += [(c, t) for t in ht + ex]
    m = pd.DataFrame(order, columns=['Country', 'Text'])
    # CEFR level as recorded in the published result files (the analysis source).
    prof = pd.concat([H[['Text', 'Proficiency']], extra[['Text', 'Proficiency']]]).drop_duplicates('Text')
    m['Proficiency'] = m.Text.map(prof.set_index('Text').Proficiency).replace({'Unknown': 'Native'})
    m['filename'] = m.Text.map(by_text.filename)
    m['prompt'] = m.filename.str.extract(r'_(PTJ|SMK)\d')[0]
    m['corpus_duplicate'] = m.Text.isin(dup_texts)
    m.insert(0, 'text_id', [f'ICNALE_{i + 1:04d}' for i in range(len(m))])
    m['pos_in_country'] = m.groupby('Country').cumcount()
    m = attach(m, 'icnale', 'Text')
    # GPTZero sample = first 50 drawn per country; positions of the 2 unscored essays are
    # checked explicitly below.
    G = read('icnale', 'gptzero')
    m['gptzero_in_sample'] = (m.pos_in_country < 50) | m.Text.isin(set(G.Text))
    feats = H.set_index('Text')[FEATS]
    for f in FEATS:
        m[f] = m.Text.map(feats[f])
    return m


def blogs():
    Z = read('blogs', 'zerogpt')
    m = Z[['Gender', 'Age', 'Text'] + FEATS].copy()
    m.insert(0, 'text_id', [f'BLOG_{i + 1:04d}' for i in range(len(m))])
    m['Age Group'] = pd.cut(m.Age, [0, 17, 27, 47, 200], labels=['13-17', '23-27', '33-47', '48+']).astype(str)
    return attach(m, 'blogs', 'Text')


def arxiv():
    frames = [read('arxiv', d)[['ArXiv_ID', 'Academic_Field', 'Text'] + FEATS] for d in FILES['arxiv']]
    m = pd.concat(frames).drop_duplicates('ArXiv_ID').reset_index(drop=True)
    m['ArXiv_ID'] = m.ArXiv_ID.astype(str)
    m.insert(0, 'text_id', [f'ARXIV_{i + 1:04d}' for i in range(len(m))])
    out = m.copy()
    for det in FILES['arxiv']:
        d = read('arxiv', det)
        d['ArXiv_ID'] = d.ArXiv_ID.astype(str)
        d = d.set_index('ArXiv_ID')
        lab = out.ArXiv_ID.map(d['Predicted Label'])
        out[f'{det}_label'] = lab
        out[f'{det}_flag'] = lab.map({'ChatGPT': 1, 'Human': 0})
        out[f'{det}_score'] = out.ArXiv_ID.map(d['Prediction Score'])
    return out


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    for name, fn in (('icnale', icnale), ('blog', blogs), ('arxiv', arxiv)):
        m = fn()
        m.to_csv(os.path.join(OUT, f'{name}_master.csv'), index=False)
        cov = {d: int(m[f'{d}_flag'].notna().sum()) for d in FILES['icnale']}
        print(name, len(m), 'scored per detector:', cov)
