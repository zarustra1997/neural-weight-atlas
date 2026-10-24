#!/usr/bin/env python3
"""Turn the pruned GGUF (prune_experts.py) into the final model and, optionally, add it to Ollama.

llama-quantize MXFP4_MOE keeps the expert tensors in MXFP4 byte for byte and moves the 16-bit rest to Q8_0. With the
llama.cpp of Ollama.app 0.40.1 (or the commit pinned in the Dockerfile), the K=20 guided cut becomes the published
gpt-oss-20b-final byte for byte; the result says whether it did. Writes results/prune/final_<label>.json.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time

import prune_experts
import reduce_installed

QTYPE = 'MXFP4_MOE'
PUBLISHED_FINAL = {  # SHA-256 do corte -> SHA-256 do modelo final publicado (blob do gpt-oss-20b-final)
    '70df4e5c31e2bca2d919e4350f12f4d745d5b9b3ef41b6866d50365b2c8317f5':
        'ae3d86441d3c04a94ab4e532879ceee449607ed1fe8bff1280ea637af32503df',
}


def quantize(source, target):
    start = time.perf_counter()
    done = subprocess.run([str(reduce_installed.QUANTIZE), str(source), str(target), QTYPE, str(reduce_installed.THREADS)],
                          capture_output=True, text=True, timeout=reduce_installed.TIMEOUT_SECONDS)
    if done.returncode:
        raise SystemExit(f'llama-quantize failed: {(done.stdout + done.stderr)[-800:]}')
    return round(time.perf_counter() - start, 1)


def run(a):
    source = prune_experts.MODELS / f'gpt-oss-20b-{a.label}.gguf'
    if not source.exists():
        raise SystemExit(f'{source.name} not found; run prune_experts.py --label {a.label} first')
    target = prune_experts.MODELS / f'gpt-oss-20b-{a.label}-final.gguf'
    source_sha = reduce_installed.sha256(source)
    seconds = quantize(source, target)
    out_sha = reduce_installed.sha256(target)
    expected = PUBLISHED_FINAL.get(source_sha)
    result = {'label': a.label, 'type': QTYPE, 'input': str(source.relative_to(prune_experts.ROOT)), 'input_sha256': source_sha,
              'output': str(target.relative_to(prune_experts.ROOT)), 'output_bytes': target.stat().st_size,
              'output_sha256': out_sha, 'published_sha256': expected,
              'identical_to_published': out_sha == expected if expected else None, 'quantize_seconds': seconds}
    if a.ollama_name:
        result['ollama_name'] = a.ollama_name
        result['ollama_create_ok'], result['ollama_create_tail'] = prune_experts.register(target, a.ollama_name)
    (prune_experts.OUT / f'final_{a.label}.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'ollama_create_tail'}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--label', required=True, help='the --label given to prune_experts.py')
    p.add_argument('--ollama-name', help='also add the final model to Ollama under this name')
    run(p.parse_args())
