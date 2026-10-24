#!/usr/bin/env python3
"""A2: count how often each gpt-oss expert is chosen (and with how much router weight) on the calibration tokens.

Runs gptoss_forward.forward() unchanged. Refuses to run unless: the forward code is the one checked
(same SHA-256 as in the validation), top-1 matched the llama-server reference in every case, routing was robust
to llama.cpp-style rounding (>= 98% of routing slots equal) and the calibration meets its minimum.
Writes results/experts/usage.json in the contract read by prune_experts.py.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import gptoss_forward as gf

np = gf.np
OUT = gf.OUT
VALIDATION = OUT / 'forward_validation_20261018.json'
ROBUSTNESS = OUT / 'routing_robustness_20261018.json'
CALIBRATION = OUT / 'calibration_tokens_v1.json'
USAGE = OUT / 'usage.json'
MIN_ROUTING_AGREEMENT = 0.98
BATCH_TOKENS = 16384
COVERAGE_K = (4, 8, 16, 20, 24, 32)


def gate():
    validation = json.loads(VALIDATION.read_text())
    robustness = json.loads(ROBUSTNESS.read_text())
    calibration = json.loads(CALIBRATION.read_text())
    problems = []
    if validation['forward_code_sha256'] != gf.file_hash(gf.__file__):
        problems.append('gptoss_forward.py changed after validation')
    if not all(c['same_top1'] for c in validation['checks']):
        problems.append('top-1 differs from the reference')
    if robustness['summary']['count_agreement'] < MIN_ROUTING_AGREEMENT:
        problems.append('routing not robust to rounding')
    if not calibration['meets_minimum'] or calibration['model_sha256'] != gf.SOURCE.name.removeprefix('sha256-'):
        problems.append('calibration below minimum or for another model')
    if problems:
        raise SystemExit('refusing to count usage: ' + '; '.join(problems))
    return validation, robustness, calibration


def batches(sequences):
    batch, size = [], 0
    for s in sequences:
        if batch and size + len(s) > BATCH_TOKENS:
            yield batch
            batch, size = [], 0
        batch.append(s)
        size += len(s)
    if batch:
        yield batch


def coverage(weight_sum):
    """Share of the router weight covered by the top-K experts of each layer (mean and worst layer)."""
    shares = {k: [] for k in COVERAGE_K}
    for w in weight_sum:
        w = np.sort(np.asarray(w))[::-1]
        for k in COVERAGE_K:
            shares[k].append(float(w[:k].sum() / w.sum()))
    return ({str(k): float(np.mean(v)) for k, v in shares.items()},
            {str(k): float(np.min(v)) for k, v in shares.items()})


def run():
    if USAGE.exists():
        raise FileExistsError(USAGE)
    validation, robustness, calibration = gate()
    sequences = calibration['token_ids']
    counts = weights = None
    started, log = time.perf_counter(), []
    for i, batch in enumerate(batches(sequences)):
        w = gf.Weights()
        try:
            _, rows = gf.forward(w, batch, lambda r: print(json.dumps({'batch': i, 'layer': r['layer'],
                                                                        'seconds': round(r['seconds'], 1),
                                                                        'rss': r['peak_rss_bytes']}), flush=True))
        finally:
            w.close()
        c = np.array([r['count'] for r in rows], np.int64)
        s = np.array([r['weight_sum'] for r in rows], np.float64)
        counts, weights = (c, s) if counts is None else (counts + c, weights + s)
        log.append({'batch': i, 'sequences': len(batch), 'tokens': sum(map(len, batch)),
                    'seconds': round(sum(r['seconds'] for r in rows), 1), 'peak_rss_bytes': max(r['peak_rss_bytes'] for r in rows)})
    mean_cov, worst_cov = coverage(weights)
    usage = {'model_sha256': calibration['model_sha256'], 'text_sha256': calibration['text_sha256'],
             'tokens': calibration['tokens'],
             'layers': [{'layer': l, 'count': counts[l].tolist(), 'weight_sum': weights[l].tolist()} for l in range(len(counts))],
             'coverage': mean_cov, 'coverage_worst_layer': worst_cov,
             'forward_code_sha256': validation['forward_code_sha256'],
             'evidence': {'validation': VALIDATION.name, 'routing_agreement': robustness['summary']['count_agreement'],
                          'calibration': CALIBRATION.name, 'code_share': calibration['code_share'],
                          'tokens_by_language': calibration['tokens_by_language']},
             'batches': log, 'seconds': round(time.perf_counter() - started, 1)}
    USAGE.write_text(json.dumps(usage, indent=1) + '\n')
    print(json.dumps({k: v for k, v in usage.items() if k != 'layers'}, indent=1))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
