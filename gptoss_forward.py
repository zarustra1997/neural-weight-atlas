#!/usr/bin/env python3
"""Bounded NumPy GPT-OSS forward candidate; validation is required before usage collection.

Weights are read with pread, one expert/matrix at a time. No model-wide mapping,
PyTorch, tokenizer download, or modification of the source. This is an experimental
implementation, not a validated replacement for Ollama. See EXPERTS_USAGE_REPORT.md.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import struct
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'NEURAL_WEIGHT_ATLAS_V1' / 'vendor'))
# Set before importing NumPy; avoid occupying all cores on the shared Mac.
BLAS_THREADS = 2
for variable in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ[variable] = str(BLAS_THREADS)
import numpy as np

SOURCE = Path.home() / '.ollama/models/blobs/sha256-8ecb5d4ad6172c63f9cbf1a07a04e525985a8a11a6e516da02b674c4010de716'
OUT = ROOT / 'results' / 'experts'
MAX_RSS_BYTES = 3_000_000_000
MAX_SEQUENCE_TOKENS = 512
MAX_TOTAL_TOKENS = 32768
DECODE_BLOCKS = 65536
OUTPUT_ROWS = 2048
ATTENTION_HEAD_BATCH = 8
SWIGLU_ALPHA = np.float32(1.702)
SWIGLU_LIMIT = np.float32(7)
TOP_K = 5
MAX_TOP5_LOGPROB_ERROR = 0.15
TYPE_LAYOUT = {0: (1, 4), 1: (1, 2), 30: (1, 2), 39: (32, 17)}
SCALAR_FORMAT = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i', 6: 'f',
                 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
MX_VALUES = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], np.float32)


def rss_check():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if rss > MAX_RSS_BYTES:
        raise MemoryError(f'RSS {rss} exceeds budget {MAX_RSS_BYTES}')
    return rss


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while data := handle.read(8 << 20):
            digest.update(data)
    return digest.hexdigest()


def read_header(path):
    """Read scalar metadata and the directory; skip tokenizer arrays without materializing them."""
    with Path(path).open('rb') as f:
        def exact(n):
            data = f.read(n)
            if len(data) != n:
                raise EOFError('truncated GGUF header')
            return data

        def num(fmt):
            return struct.unpack('<' + fmt, exact(struct.calcsize('<' + fmt)))[0]

        def string():
            n = num('Q')
            if n > (16 << 20):
                raise ValueError('unreasonable GGUF header string length')
            return exact(n).decode('utf8')

        def skip(typ):
            if typ in SCALAR_FORMAT:
                f.seek(struct.calcsize('<' + SCALAR_FORMAT[typ]), 1)
            elif typ == 8:
                f.seek(num('Q'), 1)
            elif typ == 9:
                sub, count = num('I'), num('Q')
                if sub in SCALAR_FORMAT:
                    f.seek(count * struct.calcsize('<' + SCALAR_FORMAT[sub]), 1)
                else:
                    for _ in range(count):
                        skip(sub)
            else:
                raise ValueError(f'unknown GGUF metadata type {typ}')

        if exact(4) != b'GGUF' or num('I') != 3:
            raise ValueError('expected GGUF v3')
        nt, nk = num('Q'), num('Q')
        metadata = {}
        for _ in range(nk):
            key, typ = string(), num('I')
            if typ in SCALAR_FORMAT:
                metadata[key] = num(SCALAR_FORMAT[typ])
            elif typ == 8:
                metadata[key] = string()
            else:
                skip(typ)
        tensors = {}
        for _ in range(nt):
            name, rank = string(), num('I')
            if name in tensors or not 1 <= rank <= 4:
                raise ValueError('invalid GGUF tensor directory')
            dims = [num('Q') for _ in range(rank)]
            typ, offset = num('I'), num('Q')
            if typ not in TYPE_LAYOUT or any(n <= 0 for n in dims):
                raise ValueError(f'unsupported tensor {name}: {typ} {dims}')
            block, size = TYPE_LAYOUT[typ]
            elements = math.prod(dims)
            if dims[0] % block:
                raise ValueError('partial quantization block')
            tensors[name] = {'dims': dims, 'type': typ, 'offset': offset,
                             'bytes': elements // block * size}
        alignment = metadata.get('general.alignment', 32)
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError('invalid GGUF alignment')
        start = (f.tell() + alignment - 1) // alignment * alignment
    source_size = Path(path).stat().st_size
    for t in tensors.values():
        t['offset'] += start
        if t['offset'] + t['bytes'] > source_size:
            raise EOFError('tensor extends beyond source')
    return metadata, tensors


def decode_mxfp4(raw):
    """Exact GGUF MXFP4 block expansion. Nibbles represent first16, then last16 weights."""
    raw = np.frombuffer(raw, dtype=np.uint8)
    if raw.size % 17:
        raise ValueError('partial MXFP4 block')
    blocks = raw.reshape(-1, 17)
    out = np.empty((len(blocks), 32), dtype=np.float32)
    for start in range(0, len(blocks), DECODE_BLOCKS):
        b = blocks[start:start + DECODE_BLOCKS]
        exponent = b[:, 0].astype(np.uint32)
        # IEEE subnormal exponents 0 and1 follow the official GGML half-scale convention.
        bits = np.where(exponent < 2, np.uint32(0x00200000) << exponent,
                        (exponent - np.uint32(1)) << np.uint32(23)).astype(np.uint32)
        scale = bits.view(np.float32)[:, None]
        target = out[start:start + len(b)]
        target[:, :16] = MX_VALUES[b[:, 1:] & 15] * scale
        target[:, 16:] = MX_VALUES[b[:, 1:] >> 4] * scale
    return out.reshape(-1)


def decode(raw, typ):
    if typ == 39:
        return decode_mxfp4(raw)
    if typ == 30:
        return (np.frombuffer(raw, '<u2').astype(np.uint32) << np.uint32(16)).view(np.float32)
    if typ in (0, 1):
        return np.frombuffer(raw, '<f4' if typ == 0 else '<f2').astype(np.float32, copy=True)
    raise ValueError(f'unsupported tensor type {typ}')


class Weights:
    def __init__(self, path=SOURCE):
        self.path = Path(path)
        self.metadata, self.tensors = read_header(self.path)
        if self.metadata.get('general.architecture') != 'gpt-oss':
            raise ValueError('expected original GPT-OSS architecture')
        self.fd = os.open(self.path, os.O_RDONLY)
        self.before = os.fstat(self.fd)
        self.read_bytes = 0

    def close(self):
        after = os.fstat(self.fd)
        os.close(self.fd)
        if (self.before.st_ino, self.before.st_size, self.before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
            raise RuntimeError('source changed during read')

    def raw(self, offset, length):
        if not 0 <= offset <= self.before.st_size or not 0 <= length <= self.before.st_size - offset:
            raise ValueError('read outside source')
        data = os.pread(self.fd, length, offset)
        if len(data) != length:
            raise EOFError('short model read')
        self.read_bytes += length
        return data

    def tensor(self, name, expert=None):
        t = self.tensors[name]
        dims, offset, length = t['dims'], t['offset'], t['bytes']
        if expert is not None:
            if not 0 <= expert < dims[-1]:
                raise ValueError('expert outside tensor')
            length //= dims[-1]
            offset += expert * length
            dims = dims[:-1]
        # Accidental reads of a complete expert tensor or the vocabulary matrix are forbidden.
        if math.prod(dims) * 4 > 128 << 20:
            raise MemoryError('tensor must be read by expert or row batch')
        value = decode(self.raw(offset, length), t['type']).reshape(tuple(reversed(dims)))
        rss_check()
        return value

    def rows(self, name, rows):
        t = self.tensors[name]
        if t['type'] != 30 or len(t['dims']) != 2:
            raise ValueError('expected BF16 embedding/output rows')
        width, count = t['dims']
        ids = np.asarray(rows, dtype=np.int64)
        if ids.ndim != 1 or len(ids) > OUTPUT_ROWS or np.any((ids < 0) | (ids >= count)):
            raise ValueError('invalid row selection')
        out = np.empty((len(ids), width), np.float32)
        for i, row in enumerate(ids):
            out[i] = decode(self.raw(t['offset'] + int(row) * width * 2, width * 2), 30)
        return out

    def logits(self, hidden):
        width, count = self.tensors['output.weight']['dims']
        out = np.empty((len(hidden), count), np.float32)
        for start in range(0, count, OUTPUT_ROWS):
            end = min(count, start + OUTPUT_ROWS)
            offset = self.tensors['output.weight']['offset'] + start * width * 2
            matrix = decode(self.raw(offset, (end - start) * width * 2), 30).reshape(-1, width)
            out[:, start:end] = hidden @ matrix.T
            rss_check()
        return out


def rmsnorm(x, scale, eps):
    return x * np.reciprocal(np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + np.float32(eps))) * scale


def rope_tables(length, dim, base, factor, original_context, beta_fast=32, beta_slow=1):
    """GGML YaRN convention: rounded correction dimensions, half-split (NEOX) rotation."""
    low = max(0, math.floor(dim * math.log(original_context / (2 * math.pi * beta_fast)) / (2 * math.log(base))))
    high = min(dim - 1, math.ceil(dim * math.log(original_context / (2 * math.pi * beta_slow)) / (2 * math.log(base))))
    dims = np.arange(dim // 2, dtype=np.float32)
    mask = 1 - np.clip((dims - low) / max(0.001, high - low), 0, 1)
    inv = np.power(np.float32(base), -2 * dims / np.float32(dim))
    inv *= mask + (1 - mask) / np.float32(factor)
    angle = np.arange(length, dtype=np.float32)[:, None] * inv
    magnitude = np.float32(1 + 0.1 * math.log(factor))
    return (np.cos(angle) * magnitude)[:, None, :], (np.sin(angle) * magnitude)[:, None, :]


def rotate(x, cos, sin):
    half = x.shape[-1] // 2
    a, b = x[..., :half], x[..., half:]
    return np.concatenate((a * cos - b * sin, b * cos + a * sin), axis=-1)


def attention(q, k, v, sinks, window):
    """Causal grouped attention with learned sinks; bounded head batches."""
    tokens, heads, dim = q.shape
    group = heads // k.shape[1]
    result = np.empty_like(q)
    positions = np.arange(tokens)
    allowed = positions[:, None] >= positions[None, :]
    if window:
        allowed &= positions[:, None] - positions[None, :] < window
    for start in range(0, heads, ATTENTION_HEAD_BATCH):
        hs = np.arange(start, min(heads, start + ATTENTION_HEAD_BATCH))
        qh = q[:, hs].transpose(1, 0, 2)
        kh = k[:, hs // group].transpose(1, 0, 2)
        vh = v[:, hs // group].transpose(1, 0, 2)
        scores = (qh @ kh.transpose(0, 2, 1)) * np.float32(1 / math.sqrt(dim))
        scores[:, ~allowed] = -np.inf
        peak = np.maximum(scores.max(-1, keepdims=True), sinks[hs, None, None])
        scores -= peak
        np.exp(scores, out=scores)
        scores /= scores.sum(-1, keepdims=True) + np.exp(sinks[hs, None, None] - peak)
        result[:, hs] = (scores @ vh).transpose(1, 0, 2)
    return result.reshape(tokens, heads * dim)


def route(logits, count):
    ids = np.argsort(-logits, axis=-1, kind='stable')[:, :count]
    weights = np.take_along_axis(logits, ids, axis=-1)
    weights = np.exp(weights - weights.max(-1, keepdims=True))
    weights /= weights.sum(-1, keepdims=True)
    return ids, weights


def swiglu(gate, up):
    gate = np.minimum(gate, SWIGLU_LIMIT)
    up = np.clip(up, -SWIGLU_LIMIT, SWIGLU_LIMIT)
    # exp(-logaddexp(...)) avoids overflow for large negative gates, which are not lower-clamped.
    return gate * np.exp(-np.logaddexp(np.float32(0), -SWIGLU_ALPHA * gate)) * (up + np.float32(1))


def forward(weights, sequences, progress=None):
    if not sequences or any(not 1 <= len(s) <= MAX_SEQUENCE_TOKENS for s in sequences):
        raise ValueError('sequences must contain 1..512 token IDs')
    if sum(map(len, sequences)) > MAX_TOTAL_TOKENS:
        raise MemoryError('total token budget exceeded')
    m = weights.metadata
    get = lambda suffix: m['gpt-oss.' + suffix]
    heads, kvheads = get('attention.head_count'), get('attention.head_count_kv')
    dim, eps = get('attention.key_length'), get('attention.layer_norm_rms_epsilon')
    experts, used = get('expert_count'), get('expert_used_count')
    xs = [weights.rows('token_embd.weight', s) for s in sequences]
    ropes = {n: rope_tables(n, dim, get('rope.freq_base'), get('rope.scaling.factor'),
                           get('rope.scaling.original_context_length')) for n in set(map(len, sequences))}
    rows = []
    for layer in range(get('block_count')):
        started = time.perf_counter()
        prefix = f'blk.{layer}.'
        scale = weights.tensor(prefix + 'attn_norm.weight')
        matrices = [weights.tensor(prefix + name + '.weight') for name in ('attn_q', 'attn_k', 'attn_v', 'attn_output')]
        biases = [weights.tensor(prefix + name + '.bias') for name in ('attn_q', 'attn_k', 'attn_v', 'attn_output')]
        sinks = weights.tensor(prefix + 'attn_sinks.weight')
        for i, x in enumerate(xs):
            h = rmsnorm(x, scale, eps)
            q, k, v = [h @ matrix.T + bias for matrix, bias in zip(matrices[:3], biases[:3])]
            cos, sin = ropes[len(x)]
            q = rotate(q.reshape(len(x), heads, dim), cos, sin)
            k = rotate(k.reshape(len(x), kvheads, dim), cos, sin)
            v = v.reshape(len(x), kvheads, dim)
            a = attention(q, k, v, sinks, get('attention.sliding_window') if layer % 2 == 0 else 0)
            xs[i] = x + a @ matrices[3].T + biases[3]
        del matrices, biases, q, k, v, h, a, x
        scale = weights.tensor(prefix + 'post_attention_norm.weight')
        gate = weights.tensor(prefix + 'ffn_gate_inp.weight')
        gate_bias = weights.tensor(prefix + 'ffn_gate_inp.bias')
        normalized = [rmsnorm(x, scale, eps) for x in xs]
        routed = [route(h @ gate.T + gate_bias, used) for h in normalized]
        counts, sums = np.zeros(experts, np.int64), np.zeros(experts, np.float64)
        for ids, probs in routed:
            counts += np.bincount(ids.ravel(), minlength=experts)
            sums += np.bincount(ids.ravel(), weights=probs.ravel().astype(np.float64), minlength=experts)
        # Update residuals per expert, but route only from the immutable normalized input.
        for expert in np.flatnonzero(counts):
            expert = int(expert)
            mg, mu, md = [weights.tensor(prefix + name + '_exps.weight', expert)
                          for name in ('ffn_gate', 'ffn_up', 'ffn_down')]
            bg, bu, bd = [weights.tensor(prefix + name + '_exps.bias', expert)
                          for name in ('ffn_gate', 'ffn_up', 'ffn_down')]
            for i, (h, (ids, probs)) in enumerate(zip(normalized, routed)):
                token, slot = np.nonzero(ids == expert)
                if len(token):
                    local = h[token]
                    activation = swiglu(local @ mg.T + bg, local @ mu.T + bu)
                    xs[i][token] += (activation @ md.T + bd) * probs[token, slot, None]
            del mg, mu, md, bg, bu, bd
            rss_check()
        del normalized, routed, gate, gate_bias
        if not all(np.isfinite(x).all() for x in xs):
            raise FloatingPointError(f'nonfinite hidden state in layer {layer}')
        row = {'layer': layer, 'count': counts.tolist(), 'weight_sum': sums.tolist(),
               'seconds': time.perf_counter() - started, 'peak_rss_bytes': rss_check()}
        rows.append(row)
        if progress:
            progress(row)
        gc.collect()
    scale = weights.tensor('output_norm.weight')
    return np.stack([rmsnorm(x[-1:], scale, eps)[0] for x in xs]), rows


def log_softmax(z):
    z = z.astype(np.float64)
    peak = z.max()
    return z - peak - np.log(np.exp(z - peak).sum())


def validate(reference, output):
    data = json.loads(Path(reference).read_text())
    # Fail before reading any weights if the official reference is incomplete.
    if not data.get('passed') or len(data.get('cases', [])) < 5 or not any(len(c['tokens']) == 512 for c in data['cases']):
        raise ValueError('complete official references required: >=5 cases, including 512 tokens')
    if data.get('model_sha256_from_blob_name') != SOURCE.name.removeprefix('sha256-'):
        raise ValueError('reference belongs to another model')
    if Path(output).exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    weights = Weights()
    try:
        hidden, rows = forward(weights, [c['tokens'] for c in data['cases']],
                               lambda r: print(json.dumps({'layer': r['layer'], 'seconds': r['seconds'],
                                                           'rss': r['peak_rss_bytes']}), flush=True))
        logits = weights.logits(hidden)
        checks = []
        for case, z in zip(data['cases'], logits):
            lp = log_softmax(z)
            mine = np.argsort(-lp)[:TOP_K].tolist()
            ref = case['top5']
            refids = [r['id'] for r in ref]
            error = max(abs(float(lp[r['id']]) - r['logprob']) for r in ref)
            checks.append({'name': case['name'], 'tokens': len(case['tokens']), 'numpy_top5': mine,
                           'reference_top5': refids, 'same_top1': mine[0] == refids[0],
                           'same_top5_set': set(mine) == set(refids), 'max_abs_logprob_error': error,
                           'passed': mine[0] == refids[0] and set(mine) == set(refids) and error <= MAX_TOP5_LOGPROB_ERROR})
        report = {'passed': all(c['passed'] for c in checks), 'checks': checks,
                  'model_sha256': SOURCE.name.removeprefix('sha256-'),
                  'forward_code_sha256': file_hash(__file__), 'reference_sha256': file_hash(reference),
                  'max_allowed_logprob_error': MAX_TOP5_LOGPROB_ERROR, 'peak_rss_bytes': rss_check(),
                  'seconds': time.perf_counter() - started, 'application_read_bytes': weights.read_bytes,
                  'calibration_performed': False, 'layer_timings': [r['seconds'] for r in rows]}
    finally:
        weights.close()
    Path(output).write_text(json.dumps(report, indent=2) + '\n')
    return report['passed']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', default=OUT / 'forward_validation.json', type=Path)
    args = parser.parse_args()
    return 0 if validate(args.reference, args.output) else 1


if __name__ == '__main__':
    raise SystemExit(main())
