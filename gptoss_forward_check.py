#!/usr/bin/env python3
"""Diagnose the gap between gptoss_forward.py (exact float32) and the llama-server CPU reference.

Hypothesis: llama.cpp on CPU rounds activations before each matmul (Q8_0 blocks for MXFP4 weights, BF16 for BF16
weights). This harness runs gptoss_forward.py UNCHANGED, but wraps the weight matrices so that `activation @ W.T`
first rounds the activation the way llama.cpp does. If the errors collapse, the forward's structure is right and
the gap comes from reference precision. Reads the blob only; writes results/experts/forward_check_*.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import gptoss_forward as gf

np = gf.np
Q8_BLOCK = 32
Q8_MAX = 127
MXFP4, BF16 = 39, 30


def round_bf16(x):
    """float32 -> bfloat16 -> float32, round to nearest even (GGML_FP32_TO_BF16)."""
    bits = np.ascontiguousarray(x, np.float32).view(np.uint32)
    bits = (bits + np.uint32(0x7FFF) + ((bits >> np.uint32(16)) & np.uint32(1))) & np.uint32(0xFFFF0000)
    return bits.view(np.float32)


def round_q8_0(x):
    """quantize_row_q8_0: blocks of 32, scale = amax/127 stored as fp16, q = round(x/scale)."""
    shape = x.shape
    blocks = np.ascontiguousarray(x, np.float32).reshape(-1, Q8_BLOCK)
    scale = (np.abs(blocks).max(1, keepdims=True) / Q8_MAX).astype(np.float16).astype(np.float32)
    safe = np.where(scale == 0, 1, scale)
    return (np.round(blocks / safe) * scale).reshape(shape)


class RoundedWeight(np.ndarray):
    """A weight matrix whose right-matmul rounds the activation like llama.cpp's CPU vec_dot type."""

    def __array_finalize__(self, obj):
        self.ggml_type = getattr(obj, 'ggml_type', None)

    def __rmatmul__(self, activation):
        rounded = round_q8_0(activation) if self.ggml_type == MXFP4 else round_bf16(activation)
        return np.matmul(rounded, self.view(np.ndarray))


def wrap_weights():
    plain = gf.Weights.tensor

    def tensor(self, name, expert=None):
        value = plain(self, name, expert)
        typ = self.tensors[name]['type']
        if value.ndim == 2 and typ in (MXFP4, BF16):
            value = value.view(RoundedWeight)
            value.ggml_type = typ
        return value

    def logits(self, hidden):
        width, count = self.tensors['output.weight']['dims']
        out = np.empty((len(hidden), count), np.float32)
        rounded = round_bf16(hidden)
        for start in range(0, count, gf.OUTPUT_ROWS):
            end = min(count, start + gf.OUTPUT_ROWS)
            offset = self.tensors['output.weight']['offset'] + start * width * 2
            matrix = gf.decode(self.raw(offset, (end - start) * width * 2), BF16).reshape(-1, width)
            out[:, start:end] = rounded @ matrix.T
        return out

    gf.Weights.tensor, gf.Weights.logits = tensor, logits


def check(reference, output):
    data = json.loads(Path(reference).read_text())
    if output.exists():
        raise FileExistsError(output)
    wrap_weights()
    started = time.perf_counter()
    weights = gf.Weights()
    try:
        hidden, _ = gf.forward(weights, [c['tokens'] for c in data['cases']])
        logits = weights.logits(hidden)
    finally:
        weights.close()
    checks = []
    for case, z in zip(data['cases'], logits):
        lp = gf.log_softmax(z)
        mine = np.argsort(-lp)[:gf.TOP_K].tolist()
        ref = [r['id'] for r in case['top5']]
        errors = [abs(float(lp[r['id']]) - r['logprob']) for r in case['top5']]
        checks.append({'name': case['name'], 'tokens': len(case['tokens']), 'numpy_top5': mine, 'reference_top5': ref,
                       'same_top1': mine[0] == ref[0], 'same_top5_set': set(mine) == set(ref),
                       'max_abs_logprob_error': max(errors), 'top1_logprob_error': errors[0],
                       'passed_strict_rule': mine[0] == ref[0] and set(mine) == set(ref) and max(errors) <= gf.MAX_TOP5_LOGPROB_ERROR})
    report = {'mode': 'llama.cpp CPU activation rounding emulated (Q8_0 for MXFP4, BF16 for BF16 weights)',
              'forward_code_sha256': gf.file_hash(gf.__file__), 'reference_sha256': gf.file_hash(reference),
              'passed_strict_rule': all(c['passed_strict_rule'] for c in checks), 'checks': checks,
              'seconds': round(time.perf_counter() - started, 1), 'peak_rss_bytes': gf.rss_check()}
    output.write_text(json.dumps(report, indent=1) + '\n')
    for c in checks:
        print(c['name'], c['tokens'], 'top1', c['same_top1'], 'top5', c['same_top5_set'],
              'erro', round(c['max_abs_logprob_error'], 3))
    print('regra estrita:', 'PASSOU' if report['passed_strict_rule'] else 'FALHOU', '|', report['seconds'], 's')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--reference', type=Path, default=gf.OUT / 'forward_reference_20261018.json')
    p.add_argument('--output', type=Path, default=gf.OUT / 'forward_check_emulated_20261018.json')
    a = p.parse_args()
    check(a.reference, a.output)
