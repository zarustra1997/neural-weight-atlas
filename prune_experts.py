#!/usr/bin/env python3
"""Cut experts out of a MoE GGUF (gpt-oss): keep K experts per layer and, optionally, add the result to Ollama.

The expert is the slowest dimension of every expert tensor ([..., 32]), so each kept expert is one contiguous slice
copied as is: no requantization, the kept weights are untouched. With all experts kept the output is byte-identical
to the source (checked by hashing the stream, nothing written). Every output tensor is compared byte by byte against its source slices. The source blob is only read
(SHA-256 checked before and after). Writes results/prune/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time

import reduce_installed

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'prune'
MODELS = OUT / 'models'
SOURCE_TAG = 'gpt-oss:20b'
DEFAULT_ALIGNMENT = 32
GGUF_UINT32, GGUF_STRING, GGUF_ARRAY = 4, 8, 9
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
TYPE_BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 39: (32, 17)}  # GGML type -> (weights per block, bytes per block)
EXPERT_SUFFIXES = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight',
                   'ffn_gate_exps.bias', 'ffn_up_exps.bias', 'ffn_down_exps.bias',
                   'ffn_gate_inp.weight', 'ffn_gate_inp.bias')
COPY_CHUNK = 64 << 20
TIMEOUT_SECONDS = 3600


def align(n, alignment):
    return (n + alignment - 1) // alignment * alignment


def read_header(path):
    """GGUF v3 header: raw KV bytes, where the expert_count value sits in them, tensor infos, data start."""
    with Path(path).open('rb') as f:
        u32 = lambda: struct.unpack('<I', f.read(4))[0]
        u64 = lambda: struct.unpack('<Q', f.read(8))[0]

        def skip(typ):
            if typ in SCALAR_BYTES:
                f.seek(SCALAR_BYTES[typ], 1)
            elif typ == GGUF_STRING:
                f.seek(u64(), 1)
            elif typ == GGUF_ARRAY:
                sub, count = u32(), u64()
                if sub in SCALAR_BYTES:
                    f.seek(SCALAR_BYTES[sub] * count, 1)
                else:
                    for _ in range(count):
                        skip(sub)
            else:
                raise ValueError(f'unknown GGUF value type {typ}')

        if f.read(4) != b'GGUF' or u32() != 3:
            raise ValueError('not GGUF v3')
        n_tensors, n_kv = u64(), u64()
        kv_start, alignment, arch, counts, used = f.tell(), DEFAULT_ALIGNMENT, None, {}, None
        for _ in range(n_kv):
            key, typ = f.read(u64()).decode(), u32()
            if key == 'general.alignment' and typ == GGUF_UINT32:
                alignment = u32()
            elif key == 'general.architecture' and typ == GGUF_STRING:
                arch = f.read(u64()).decode()
            elif key.endswith('.expert_count') and typ == GGUF_UINT32:
                counts[key] = f.tell() - kv_start
                f.seek(4, 1)
            elif key.endswith('.expert_used_count') and typ == GGUF_UINT32:
                used = u32()
            else:
                skip(typ)
        kv_end = f.tell()
        tensors = []
        for _ in range(n_tensors):
            name = f.read(u64()).decode()
            dims = [u64() for _ in range(u32())]
            tensors.append({'name': name, 'dims': dims, 'type': u32(), 'offset': u64()})
        data_start = align(f.tell(), alignment)
        f.seek(kv_start)
        kv = f.read(kv_end - kv_start)
    count_pos = counts[f'{arch}.expert_count']
    return {'n_kv': n_kv, 'kv': kv, 'count_pos': count_pos, 'experts': struct.unpack_from('<I', kv, count_pos)[0],
            'experts_used': used, 'alignment': alignment, 'tensors': tensors, 'data_start': data_start}


def nbytes(tensor):
    block, size = TYPE_BLOCK[tensor['type']]
    elements = math.prod(tensor['dims'])
    if elements % block:
        raise ValueError(f"{tensor['name']}: {elements} weights is not a whole number of blocks")
    return elements // block * size


def layer_of(name):
    return int(name.split('.')[1]) if name.startswith('blk.') else None


def is_expert_tensor(tensor, experts):
    return tensor['name'].startswith('blk.') and tensor['name'].endswith(EXPERT_SUFFIXES) and tensor['dims'][-1] == experts


def plan(header, keep):
    """Output tensors: new dims and the (source offset, length) pieces that make up each one, in source data order."""
    experts, start = header['experts'], header['data_start']
    out, cursor = [], 0
    for t in sorted(header['tensors'], key=lambda t: t['offset']):
        size, src = nbytes(t), start + t['offset']
        if is_expert_tensor(t, experts):
            per = size // experts
            ids = keep[layer_of(t['name'])]
            pieces, dims = [(src + e * per, per) for e in ids], t['dims'][:-1] + [len(ids)]
        else:
            pieces, dims = [(src, size)], t['dims']
        cursor = align(cursor, header['alignment'])
        out.append({'name': t['name'], 'dims': dims, 'type': t['type'], 'offset': cursor, 'pieces': pieces})
        cursor += sum(n for _, n in pieces)
    order = {t['name']: i for i, t in enumerate(header['tensors'])}
    return sorted(out, key=lambda t: order[t['name']])


def header_bytes(header, tensors, kept):
    kv = bytearray(header['kv'])
    struct.pack_into('<I', kv, header['count_pos'], kept)
    parts = [b'GGUF', struct.pack('<IQQ', 3, len(tensors), header['n_kv']), bytes(kv)]
    for t in tensors:
        name = t['name'].encode()
        parts += [struct.pack('<Q', len(name)), name, struct.pack('<I', len(t['dims'])),
                  struct.pack(f"<{len(t['dims'])}Q", *t['dims']), struct.pack('<IQ', t['type'], t['offset'])]
    raw = b''.join(parts)
    return raw + bytes(align(len(raw), header['alignment']) - len(raw))


def read_exact(fd, n, offset):
    data = os.pread(fd, n, offset)
    if len(data) != n:
        raise EOFError(f'wanted {n} bytes at {offset}, got {len(data)}')
    return data


def copy_pieces(src_fd, pieces, out):
    for offset, length in pieces:
        done = 0
        while done < length:
            n = min(COPY_CHUNK, length - done)
            out.write(read_exact(src_fd, n, offset + done))
            done += n


class HashSink:
    """Stands in for the output file: hashes the stream instead of writing it (identity check without 13.8 GB on disk)."""

    def __init__(self):
        self.digest, self.size = hashlib.sha256(), 0

    def write(self, data):
        self.digest.update(data)
        self.size += len(data)

    def tell(self):
        return self.size


def stream_pruned(source, out, header, keep):
    tensors = plan(header, keep)
    head = header_bytes(header, tensors, len(keep[0]))
    fd = os.open(source, os.O_RDONLY)
    try:
        out.write(head)
        for t in sorted(tensors, key=lambda t: t['offset']):
            out.write(bytes(len(head) + t['offset'] - out.tell()))
            copy_pieces(fd, t['pieces'], out)
    finally:
        os.close(fd)
    return tensors


def verify(source, target, tensors):
    """Re-read the output: header parses, expert_count matches, every tensor equals its source slices byte by byte."""
    got = read_header(target)
    by_name = {t['name']: t for t in got['tensors']}
    src_fd, out_fd = os.open(source, os.O_RDONLY), os.open(target, os.O_RDONLY)
    try:
        for t in tensors:
            g = by_name[t['name']]
            if g['dims'] != t['dims'] or g['type'] != t['type'] or g['offset'] != t['offset']:
                return False, f"{t['name']}: header mismatch"
            at = got['data_start'] + g['offset']
            for offset, length in t['pieces']:
                done = 0
                while done < length:
                    n = min(COPY_CHUNK, length - done)
                    if read_exact(src_fd, n, offset + done) != read_exact(out_fd, n, at + done):
                        return False, f"{t['name']}: bytes differ"
                    done += n
                at += length
    finally:
        os.close(src_fd)
        os.close(out_fd)
    return True, 'every tensor equals its source slices'


def checked_usage_rows(usage, digest, experts, layers):
    """usage.json rows, refused unless they belong to this blob and cover every layer once with finite values."""
    data = json.loads(Path(usage).read_text())
    if data.get('model_sha256') != digest:
        raise SystemExit(f'{usage}: model_sha256 is not the source blob; refusing')
    rows = sorted(data['layers'], key=lambda r: r['layer'])
    if [r['layer'] for r in rows] != list(range(layers)):
        raise SystemExit(f'{usage}: layers must be 0..{layers - 1}, each exactly once')
    for r in rows:
        values = list(r['count']) + list(r['weight_sum'])
        if len(r['count']) != experts or len(r['weight_sum']) != experts or \
                not all(isinstance(v, (int, float)) and math.isfinite(v) and v >= 0 for v in values):
            raise SystemExit(f"{usage}: layer {r['layer']} needs {experts} finite non-negative count and weight_sum values")
    return rows


def choose_keep(header, layers, digest, first=None, usage=None, k=None):
    """Kept expert ids per layer: all, the first K (arbitrary, memory test only) or top-K by router weight (usage.json).
    keep[layer][i] is the ORIGINAL id of new expert i in that layer."""
    experts, used = header['experts'], header['experts_used']
    wanted = k if usage else first
    if wanted is not None and not used <= wanted <= experts:
        raise SystemExit(f'K must be between {used} (experts used per token) and {experts}')
    if usage:
        rows = checked_usage_rows(usage, digest, experts, layers)
        return 'usage-top-k', [sorted(sorted(range(experts), key=lambda e: (-r['weight_sum'][e], -r['count'][e], e))[:k])
                               for r in rows]
    if first:
        return 'first-k-arbitrary', [list(range(first))] * layers
    return 'all', [list(range(experts))] * layers


def register(target, name):
    modelfile_text = subprocess.run(['ollama', 'show', '--modelfile', SOURCE_TAG], capture_output=True, text=True,
                                    check=True).stdout
    lines = modelfile_text.splitlines()
    while lines and lines[0].startswith('#'):  # only the comment block on top; the gpt-oss template has '# ' lines
        lines.pop(0)
    modelfile = OUT / f'Modelfile.{name}'
    modelfile.write_text('\n'.join(f'FROM {target}' if l.startswith('FROM ') else l for l in lines) + '\n')
    done = subprocess.run(['ollama', 'create', name, '-f', str(modelfile)], capture_output=True, text=True,
                          timeout=TIMEOUT_SECONDS)
    return done.returncode == 0, (done.stdout + done.stderr)[-300:]


def run(a):
    MODELS.mkdir(parents=True, exist_ok=True)
    source, digest = reduce_installed.blob_of(SOURCE_TAG)
    before = reduce_installed.sha256(source)
    if before != digest:
        raise SystemExit('source blob does not match its own digest; stopping')
    header = read_header(source)
    layers = 1 + max(layer_of(t['name']) for t in header['tensors'] if t['name'].startswith('blk.'))
    rule, keep = choose_keep(header, layers, digest, a.first, a.usage, a.k)
    if rule == 'all':
        sink = HashSink()
        stream_pruned(source, sink, header, keep)
        same = sink.digest.hexdigest() == before
        check = {'check': 'keep all experts -> same bytes as the source', 'source_sha256': before,
                 'stream_sha256': sink.digest.hexdigest(), 'identical_to_source': same, 'stream_bytes': sink.size,
                 'source_bytes': source.stat().st_size}
        (OUT / f'prune_{a.label}.json').write_text(json.dumps(check, indent=1) + '\n')
        print(json.dumps(check, indent=1))
        return
    target = MODELS / f'gpt-oss-20b-{a.label}.gguf'
    start = time.perf_counter()
    with target.open('wb') as out:
        tensors = stream_pruned(source, out, header, keep)
    write_seconds = time.perf_counter() - start
    ok, detail = verify(source, target, tensors)
    out_sha = reduce_installed.sha256(target)
    result = {'label': a.label, 'source_tag': SOURCE_TAG, 'source_blob': str(source), 'source_bytes': source.stat().st_size,
              'source_sha256_before': before, 'rule': rule, 'experts_kept_per_layer': len(keep[0]), 'keep': keep,
              'keep_meaning': 'keep[layer][i] = original id of new expert i',
              'usage_file': a.usage, 'output': str(target), 'output_bytes': target.stat().st_size,
              'vs_source': target.stat().st_size / source.stat().st_size, 'output_sha256': out_sha,
              'verify_ok': ok, 'verify_detail': detail,
              'write_seconds': round(write_seconds, 1)}
    if ok and a.ollama_name:
        result['ollama_name'] = a.ollama_name
        result['ollama_create_ok'], result['ollama_create_tail'] = register(target, a.ollama_name)
    result['source_sha256_after'] = reduce_installed.sha256(source)
    result['source_unchanged'] = result['source_sha256_after'] == before == digest
    (OUT / f'prune_{a.label}.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('keep', 'ollama_create_tail')}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True)
    p.add_argument('--first', type=int, help='keep experts 0..K-1 in every layer (arbitrary: memory/speed test only)')
    p.add_argument('--usage', help='usage.json (contract): keep the top-K experts by router weight per layer')
    p.add_argument('--k', type=int, help='experts to keep per layer with --usage')
    p.add_argument('--ollama-name', help='also add the output to Ollama under this test name')
    a = p.parse_args()
    if a.usage and not a.k:
        p.error('--usage needs --k')
    run(a)
