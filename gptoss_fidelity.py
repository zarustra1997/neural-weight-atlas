#!/usr/bin/env python3
"""P1 (NWA_V4_PROVAS.md): how much does cutting experts change gpt-oss's next-token distribution?

Text the cut never saw: the 150 conversations the ORIGINAL model produced in the tests (question, reasoning and
answer, harmony format) plus local pt/en prose not used in the calibration. Variants: original (32 experts),
K=20 guided (the final model), K=16 guided and K=20 random (controls). A cut is applied by blocking the removed
experts in the router (bias -inf), which is what the pruned file does; attention/output stay BF16 in all variants,
so only the cut is measured. Reuses the validated functions of gptoss_forward.py (unchanged); this file only
re-runs the same layer loop returning every position, and checks itself against gptoss_forward.forward first.
Writes results/v4/fidelity*.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import subprocess
import time

import gptoss_forward as gf
import bench120
import gptoss_calibration as cal
import reference_topk as ref

np = gf.np
OUT = gf.ROOT / 'results' / 'v4'
TOKENS = OUT / 'fidelity_tokens.json'
REPORT = OUT / 'fidelity.json'
HOLDOUT = gf.ROOT / 'results' / 'holdout' / 'holdout_H01_H30.json'
ANSWERS = (gf.ROOT / 'results' / 'moe' / 'answers_gpt-oss-20b.json', gf.ROOT / 'results' / 'holdout' / 'answers_original.json')
PROSE = {'pt': [gf.ROOT / n for n in ('MICRODOT_REPORT.md', 'GPS_REPORT.md', 'AUDIT_BENCH120.md')],
         'en': [gf.ROOT.parent.parent / 'n8n-master' / 'AGENTS.md',
                gf.ROOT.parent.parent / 'timesfm-master (terceiro, baixável) 094957' / 'v1' / 'README.md']}
PROSE_TOKENS_PER_LANGUAGE = 2500
SEQUENCE_TOKENS = 512
BATCH_TOKENS = 16384
SAMPLED_POSITIONS = 3000
SEED = 20261020
KEEP = {'K20_guiado': gf.ROOT / 'results' / 'prune' / 'prune_k20-guiado.json',
        'K16_guiado': gf.ROOT / 'results' / 'prune' / 'prune_k16-guiado.json'}
RANDOM_K = 20
SELF_CHECK_TOKENS = 24


# ---------- data ----------
def conversations():
    questions = {q['id']: q['question'] for q in json.loads((bench120.OUT / 'questions.json').read_text())}
    questions |= {h['id']: h['question'] for h in json.loads(HOLDOUT.read_text())}
    rows = []
    for path in ANSWERS:
        for r in json.loads(path.read_text()):
            rows.append({'text': questions[r['id']], 'thinking': r.get('thinking', ''), 'answer': r.get('answer', '')})
    return rows


def prefix(item):
    return (f"<|start|>system<|message|>{cal.SYSTEM}<|end|><|start|>user<|message|>{item['text']}<|end|>"
            f"<|start|>assistant<|channel|>analysis<|message|>")


def build_tokens():
    sequences = []
    process = subprocess.Popen(ref.command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    guard = ref.Guard(process)
    guard.start()
    try:
        ref.wait_ready(process)
        for item in conversations():
            full = cal.tokenize_special(cal.harmony(item))[:SEQUENCE_TOKENS]
            generated_from = len(cal.tokenize_special(prefix(item)))
            sequences.append({'source': 'conversa_do_original', 'tokens': full, 'generated_from': generated_from})
        for lang, paths in PROSE.items():
            budget = PROSE_TOKENS_PER_LANGUAGE
            for path in paths:
                if budget <= 0:
                    break
                tokens = ref.tokenize(ref.plain_text(path))[:budget]
                budget -= len(tokens)
                for i in range(0, len(tokens), SEQUENCE_TOKENS):
                    chunk = tokens[i:i + SEQUENCE_TOKENS]
                    if len(chunk) > 8:
                        sequences.append({'source': f'prosa_{lang}:{path.name}', 'tokens': chunk, 'generated_from': None})
    finally:
        guard.done.set()
        process.terminate()
        process.wait(timeout=30)
    TOKENS.write_text(json.dumps({'model_sha256': ref.DIGEST, 'sequences': sequences}) + '\n')
    print(json.dumps({'sequences': len(sequences), 'tokens': sum(len(s['tokens']) for s in sequences)}))


# ---------- forward over all positions, with an optional cut ----------
def forward_all(weights, sequences, removed=None, keep_rows=None):
    """Same layer loop as gptoss_forward.forward; returns the final-normed hidden state of the rows in keep_rows
    (per sequence) instead of only the last one. removed: {layer: set(expert ids)} blocked in the router."""
    m = weights.metadata
    get = lambda suffix: m['gpt-oss.' + suffix]
    heads, kvheads = get('attention.head_count'), get('attention.head_count_kv')
    dim, eps = get('attention.key_length'), get('attention.layer_norm_rms_epsilon')
    experts, used = get('expert_count'), get('expert_used_count')
    xs = [weights.rows('token_embd.weight', s) for s in sequences]
    ropes = {n: gf.rope_tables(n, dim, get('rope.freq_base'), get('rope.scaling.factor'),
                               get('rope.scaling.original_context_length')) for n in set(map(len, sequences))}
    for layer in range(get('block_count')):
        prefix_name = f'blk.{layer}.'
        scale = weights.tensor(prefix_name + 'attn_norm.weight')
        matrices = [weights.tensor(prefix_name + n + '.weight') for n in ('attn_q', 'attn_k', 'attn_v', 'attn_output')]
        biases = [weights.tensor(prefix_name + n + '.bias') for n in ('attn_q', 'attn_k', 'attn_v', 'attn_output')]
        sinks = weights.tensor(prefix_name + 'attn_sinks.weight')
        for i, x in enumerate(xs):
            h = gf.rmsnorm(x, scale, eps)
            q, k, v = [h @ mat.T + b for mat, b in zip(matrices[:3], biases[:3])]
            cos, sin = ropes[len(x)]
            q = gf.rotate(q.reshape(len(x), heads, dim), cos, sin)
            k = gf.rotate(k.reshape(len(x), kvheads, dim), cos, sin)
            v = v.reshape(len(x), kvheads, dim)
            a = gf.attention(q, k, v, sinks, get('attention.sliding_window') if layer % 2 == 0 else 0)
            xs[i] = x + a @ matrices[3].T + biases[3]
        scale = weights.tensor(prefix_name + 'post_attention_norm.weight')
        gate = weights.tensor(prefix_name + 'ffn_gate_inp.weight')
        gate_bias = weights.tensor(prefix_name + 'ffn_gate_inp.bias').copy()
        if removed and removed.get(layer):
            gate_bias[sorted(removed[layer])] = -np.inf
        normalized = [gf.rmsnorm(x, scale, eps) for x in xs]
        routed = [gf.route(h @ gate.T + gate_bias, used) for h in normalized]
        counts = np.zeros(experts, np.int64)
        for ids, _ in routed:
            counts += np.bincount(ids.ravel(), minlength=experts)
        for expert in np.flatnonzero(counts):
            expert = int(expert)
            mg, mu, md = [weights.tensor(prefix_name + n + '_exps.weight', expert) for n in ('ffn_gate', 'ffn_up', 'ffn_down')]
            bg, bu, bd = [weights.tensor(prefix_name + n + '_exps.bias', expert) for n in ('ffn_gate', 'ffn_up', 'ffn_down')]
            for i, (h, (ids, probs)) in enumerate(zip(normalized, routed)):
                token, slot = np.nonzero(ids == expert)
                if len(token):
                    local = h[token]
                    xs[i][token] += (gf.swiglu(local @ mg.T + bg, local @ mu.T + bu) @ md.T + bd) * probs[token, slot, None]
            del mg, mu, md, bg, bu, bd
            gf.rss_check()
        if not all(np.isfinite(x).all() for x in xs):
            raise FloatingPointError(f'nonfinite hidden state in layer {layer}')
    scale = weights.tensor('output_norm.weight')
    rows = keep_rows if keep_rows is not None else [[len(x) - 1] for x in xs]
    return np.concatenate([gf.rmsnorm(x[r], scale, eps) for x, r in zip(xs, rows)])


def self_check(weights, sequences):
    """My loop with no cut must give the same last-position states as gptoss_forward.forward."""
    seqs = [s[:SELF_CHECK_TOKENS] for s in sequences[:2]]
    mine = forward_all(weights, seqs)
    codex, _ = gf.forward(weights, seqs)
    diff = float(np.max(np.abs(mine - codex)))
    if diff > 1e-5:
        raise SystemExit(f'self-check failed: max difference {diff}')
    return diff


# ---------- metrics ----------
def cuts(experts, layers):
    out = {}
    for label, path in KEEP.items():
        keep = json.loads(path.read_text())['keep']
        out[label] = {l: set(range(experts)) - set(keep[l]) for l in range(layers)}
    out['K20_aleatorio'] = {l: set(range(experts)) - set(random.Random(SEED + l).sample(range(experts), RANDOM_K)) for l in range(layers)}
    return out


def sample_rows(sequences, rng):
    """SAMPLED_POSITIONS positions (seeded), each predicting the next token, so the last position is excluded."""
    pool = [(i, p) for i, s in enumerate(sequences) for p in range(len(s['tokens']) - 1)]
    picks = sorted(rng.sample(pool, min(SAMPLED_POSITIONS, len(pool))))
    rows = [[] for _ in sequences]
    for i, p in picks:
        rows[i].append(p)
    generated = [p >= (sequences[i]['generated_from'] or 10 ** 9) for i, p in picks]
    return rows, picks, generated


def run_variant(weights, sequences, rows, removed):
    states = []
    batch, batch_rows, size = [], [], 0
    for s, r in zip(sequences, rows):
        if not r:
            continue
        if batch and size + len(s['tokens']) > BATCH_TOKENS:
            states.append(forward_all(weights, batch, removed, batch_rows))
            batch, batch_rows, size = [], [], 0
        batch.append(s['tokens'])
        batch_rows.append(r)
        size += len(s['tokens'])
    if batch:
        states.append(forward_all(weights, batch, removed, batch_rows))
    return np.concatenate(states)


def compare(weights, hidden):
    """One pass over the vocabulary: online log-sum-exp per variant and KL(original || variant), plus top-1."""
    labels = list(hidden)
    n = len(hidden['original'])
    width, count = weights.tensors['output.weight']['dims']
    mo, so = np.full(n, -np.inf), np.zeros(n)
    stats = {v: {'m': np.full(n, -np.inf), 's': np.zeros(n), 't': np.zeros(n), 'best': np.full(n, -np.inf),
                 'arg': np.zeros(n, np.int64)} for v in labels}
    for start in range(0, count, gf.OUTPUT_ROWS):
        end = min(count, start + gf.OUTPUT_ROWS)
        offset = weights.tensors['output.weight']['offset'] + start * width * 2
        matrix = gf.decode(weights.raw(offset, (end - start) * width * 2), 30).reshape(-1, width)
        z = {v: (hidden[v] @ matrix.T).astype(np.float64) for v in labels}
        zo = z['original']
        new_mo = np.maximum(mo, zo.max(1))
        rescale = np.exp(mo - new_mo)
        eo = np.exp(zo - new_mo[:, None])
        for v in labels:
            st = stats[v]
            st['t'] = st['t'] * rescale + (eo * (zo - z[v])).sum(1)
            new_m = np.maximum(st['m'], z[v].max(1))
            st['s'] = st['s'] * np.exp(st['m'] - new_m) + np.exp(z[v] - new_m[:, None]).sum(1)
            st['m'] = new_m
            idx = z[v].argmax(1)
            val = z[v][np.arange(n), idx]
            better = val > st['best']
            st['best'][better], st['arg'][better] = val[better], idx[better] + start
        so = so * rescale + eo.sum(1)
        mo = new_mo
        gf.rss_check()
    lse_o = mo + np.log(so)
    out = {}
    for v in labels:
        st = stats[v]
        kl = st['t'] / so - lse_o + (st['m'] + np.log(st['s']))
        out[v] = {'kl': np.maximum(kl, 0), 'top1': st['arg']}
    return out


def summarize(result, generated):
    gen = np.array(generated)
    top_o = result['original']['top1']
    rows = {}
    for v, r in result.items():
        if v == 'original':
            continue
        same = r['top1'] == top_o
        rows[v] = {'kl_mean_nats': float(r['kl'].mean()), 'kl_median': float(np.median(r['kl'])),
                   'kl_p95': float(np.percentile(r['kl'], 95)), 'top1_agreement': float(same.mean()),
                   'generated_only': {'positions': int(gen.sum()), 'kl_mean_nats': float(r['kl'][gen].mean()) if gen.any() else None,
                                      'top1_agreement': float(same[gen].mean()) if gen.any() else None}}
    final = rows['K20_guiado']
    ok = (final['kl_mean_nats'] <= 0.05 and final['top1_agreement'] >= 0.95 and
          final['kl_mean_nats'] * 2 <= min(rows['K16_guiado']['kl_mean_nats'], rows['K20_aleatorio']['kl_mean_nats']))
    return rows, ('FIDELIDADE ALTA COMPROVADA' if ok else 'NÃO ATINGIU O CRITÉRIO')


def measure():
    data = json.loads(TOKENS.read_text())
    sequences = data['sequences']
    rng = random.Random(SEED)
    rows, picks, generated = sample_rows(sequences, rng)
    started = time.perf_counter()
    weights = gf.Weights()
    try:
        diff = self_check(weights, [s['tokens'] for s in sequences])
        print(json.dumps({'self_check_max_diff': diff}), flush=True)
        variants = {'original': None, **cuts(32, 24)}
        hidden = {}
        for label, removed in variants.items():
            t = time.perf_counter()
            hidden[label] = run_variant(weights, sequences, rows, removed)
            print(json.dumps({'variant': label, 'seconds': round(time.perf_counter() - t, 1)}), flush=True)
        result = compare(weights, hidden)
    finally:
        weights.close()
    summary, verdict = summarize(result, generated)
    report = {'preregistration': 'NWA_V4_PROVAS.md P1', 'model_sha256': data['model_sha256'], 'self_check_max_diff': diff,
              'sequences': len(sequences), 'tokens': sum(len(s['tokens']) for s in sequences), 'positions': len(picks),
              'generated_positions': int(sum(generated)), 'variants': summary, 'verdict': verdict,
              'seconds': round(time.perf_counter() - started, 1), 'peak_rss_bytes': gf.rss_check()}
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=('tokens', 'measure'))
    {'tokens': build_tokens, 'measure': measure}[p.parse_args().step]()
