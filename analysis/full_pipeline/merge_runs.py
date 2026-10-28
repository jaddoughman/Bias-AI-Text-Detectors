"""Assemble complete standard-mode outputs from the pieces produced by runs/queue2.sh.

For Hello-SimpleAI and Fakespot the standard (512-token) and original-wrapper outputs are
identical on every text the original wrapper could score (ICNALE: 2,078/2,078 labels,
max |dP| = 7e-5), so the original-wrapper outputs are reused for those texts and only the
texts that failed under the original wrapper (>512 tokens) were re-scored in standard mode.
For the whitespace experiment, only essays whose text changes under normalisation were
re-scored; unchanged essays take their standard-mode outputs.
"""
import os

import pandas as pd

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs')
D = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')


def cols(d):
    return [f'{d}_p_ai', f'{d}_n_tokens', f'{d}_pred_ai', f'{d}_error']


def corpus(ds):
    orig = pd.read_csv(os.path.join(R, f'{ds}_original.csv'))
    new = pd.read_csv(os.path.join(R, f'{ds}_std_new.csv'))
    long_ = pd.read_csv(os.path.join(R, f'{ds}_std_long.csv')).set_index('text_id')
    out = orig[['text_id']].copy()
    for d in ['hello', 'fakespot']:
        o = orig.set_index('text_id')[cols(d)].copy()
        failed = o[f'{d}_error'].fillna('').ne('')
        for c in (f'{d}_p_ai', f'{d}_n_tokens'):
            o[c] = o[c].astype(float)
            o.loc[failed, c] = long_.loc[o.index[failed], c].astype(float).values
        o[f'{d}_error'] = ''
        o[f'{d}_pred_ai'] = o[f'{d}_p_ai'] > 0.5
        out = out.join(o, on='text_id')
    assert (new.text_id.values == out.text_id.values).all()
    for d in ['roberta_large', 'radar', 'e5']:
        out[cols(d)] = new[cols(d)].values
    out.to_csv(os.path.join(R, f'{ds}_standard.csv'), index=False)
    print(ds, len(out), {d: int(out[f'{d}_pred_ai'].astype(bool).sum()) for d in ['hello', 'fakespot', 'roberta_large', 'radar', 'e5']})


def whitespace():
    std = pd.read_csv(os.path.join(R, 'icnale_standard.csv')).set_index('text_id')
    ch = pd.read_csv(os.path.join(R, 'icnale_ws_changed.csv')).set_index('text_id')
    out = std.copy()
    for d in ['hello', 'fakespot', 'roberta_large', 'radar']:
        for c in (f'{d}_p_ai', f'{d}_n_tokens'):
            out[c] = out[c].astype(float)
            out.loc[ch.index, c] = ch[c].astype(float)
        out[f'{d}_pred_ai'] = out[f'{d}_p_ai'] > 0.5
    out = out[[c for d in ['hello', 'fakespot', 'roberta_large', 'radar'] for c in cols(d)]]
    out.reset_index().to_csv(os.path.join(R, 'icnale_ws_standard.csv'), index=False)
    print('whitespace merged', len(out), 'changed', len(ch))


if __name__ == '__main__':
    import sys
    for a in sys.argv[1:]:
        corpus(a) if a in ('blog', 'arxiv') else whitespace()
