#!/usr/bin/env python3
"""Reduce a model already in the local Ollama with the llama-quantize bundled in Ollama.app, then add it back to
Ollama under a test name with the original's Modelfile (only FROM changes). The original blob is only read
(SHA-256 checked before and after). Writes results/moe/models/ and results/moe/reduce_<name>.json.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'moe'
MODELS = OUT / 'models'
QUANTIZE = Path(os.environ.get('LLAMA_QUANTIZE', '/Applications/Ollama.app/Contents/Resources/llama-quantize'))
OLLAMA_ROOT = Path(os.environ.get('OLLAMA_MODELS', Path.home() / '.ollama' / 'models'))  # mesmo nome que o Ollama usa
MANIFEST_DIRS = ('manifests/registry.ollama.ai/library', 'manifests-v2/ollama.com/library')  # layout antigo e novo
THREADS = 8
TIMEOUT_SECONDS = 3600


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1 << 24), b''):
            h.update(chunk)
    return h.hexdigest()


def blob_of(tag):
    name, _, version = tag.partition(':')
    paths = [OLLAMA_ROOT / d / name / (version or 'latest') for d in MANIFEST_DIRS]
    found = next((m for m in paths if m.exists()), None)
    if found is None:
        raise SystemExit(f'{tag}: not in Ollama ({OLLAMA_ROOT}); run: ollama pull {tag}')
    manifest = json.loads(found.read_text())
    digest = next(l['digest'] for l in manifest['layers'] if l['mediaType'] == 'application/vnd.ollama.image.model')
    return OLLAMA_ROOT / 'blobs' / digest.replace(':', '-'), digest.split(':', 1)[1]


def run(tag, qtype, new_name):
    MODELS.mkdir(parents=True, exist_ok=True)
    source, digest = blob_of(tag)
    before = sha256(source)
    if before != digest:
        raise SystemExit('source blob does not match its own digest; stopping')
    target = MODELS / f"{tag.replace(':', '_')}-{qtype}.gguf"
    start = time.perf_counter()
    done = subprocess.run([str(QUANTIZE), '--allow-requantize', str(source), str(target), qtype, str(THREADS)],
                          capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    seconds = time.perf_counter() - start
    if done.returncode:
        raise SystemExit(f'llama-quantize failed: {(done.stdout + done.stderr)[-800:]}')
    modelfile_text = subprocess.run(['ollama', 'show', '--modelfile', tag], capture_output=True, text=True, check=True).stdout
    lines = modelfile_text.splitlines()
    while lines and lines[0].startswith('#'):  # only the comment block on top; the gpt-oss template has '# ' lines
        lines.pop(0)
    lines = [f'FROM {target}' if l.startswith('FROM ') else l for l in lines]
    modelfile = OUT / f'Modelfile.{new_name}'
    modelfile.write_text('\n'.join(lines) + '\n')
    created = subprocess.run(['ollama', 'create', new_name, '-f', str(modelfile)], capture_output=True, text=True, timeout=TIMEOUT_SECONDS)
    after = sha256(source)
    result = {'source_tag': tag, 'source_blob': str(source), 'source_bytes': source.stat().st_size,
              'source_sha256_before': before, 'source_sha256_after': after, 'source_unchanged': before == after == digest,
              'type': qtype, 'mode': '--allow-requantize from the installed quantization (no 16-bit original here)',
              'output': str(target), 'output_bytes': target.stat().st_size, 'vs_source': target.stat().st_size / source.stat().st_size,
              'output_sha256': sha256(target), 'quantize_seconds': round(seconds, 1), 'ollama_name': new_name,
              'ollama_create_ok': created.returncode == 0, 'ollama_create_tail': (created.stdout + created.stderr)[-300:]}
    (OUT / f'reduce_{new_name}.json').write_text(json.dumps(result, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'ollama_create_tail'}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tag')
    p.add_argument('--type', default='Q3_K_S')
    p.add_argument('--name', required=True)
    a = p.parse_args()
    run(a.tag, a.type, a.name)
