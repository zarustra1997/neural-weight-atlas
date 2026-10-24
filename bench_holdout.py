#!/usr/bin/env python3
"""Independent holdout (H01-H30) on gpt-oss models in Ollama, scored like bench120 (strict first line).

The holdout file is used only if its SHA-256 is the pinned one.
Same settings as bench_moe.py (temperature 0, fixed seed, reasoning low). Writes results/holdout/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

import bench120
import bench_moe

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'holdout'
SOURCE = ROOT / 'results' / 'holdout' / 'holdout_H01_H30.json'
COPY = OUT / 'holdout_H01_H30.json'
SHA256 = '7b14bf5f28de6f1a4e309ba32aa9f6a415d2e01e1342c35786195787960c2d61'


def questions():
    OUT.mkdir(parents=True, exist_ok=True)
    if not COPY.exists():
        data = SOURCE.read_bytes()
        if hashlib.sha256(data).hexdigest() != SHA256:
            raise SystemExit('holdout file does not match its pinned SHA-256; stopping')
        COPY.write_bytes(data)
    if hashlib.sha256(COPY.read_bytes()).hexdigest() != SHA256:
        raise SystemExit('holdout copy changed; stopping')
    return json.loads(COPY.read_text())


def run(model, label):
    rows = []
    bench_moe.ask_ollama(model, 'Responda só: ok', 16)  # load, not scored
    for q in questions():
        r = bench_moe.ask_ollama(model, q['question'], bench_moe.ANSWER_TOKENS)
        rows.append({'id': q['id'], **r, 'exact': bench120.exact(r['answer'], q['accept'], q.get('case_sensitive', False)),
                     'empty_answer': not r['answer'].strip()})
    subprocess.run(['ollama', 'stop', model], capture_output=True)  # one model in RAM at a time
    k = sum(r['exact'] for r in rows)
    summary = {'label': label, 'model': model, 'holdout_sha256': SHA256, 'exact': k, 'n': len(rows),
               'wilson_95': bench120.wilson(k, len(rows)), 'empty_answers': sum(r['empty_answer'] for r in rows),
               'wrong_ids': [r['id'] for r in rows if not r['exact']]}
    (OUT / f'answers_{label}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n')
    (OUT / f'summary_{label}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(summary, ensure_ascii=False))
    return [r['exact'] for r in rows]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', action='append', required=True, help='name=label, e.g. gpt-oss:20b=original')
    pairs = [m.rsplit('=', 1) for m in p.parse_args().model]
    results = {label: run(model, label) for model, label in pairs}
    labels = list(results)
    if len(labels) == 2:
        comparison = bench120.mcnemar(results[labels[0]], results[labels[1]])
        (OUT / f'compare_{labels[0]}_vs_{labels[1]}.json').write_text(json.dumps(comparison, indent=1) + '\n')
        print(labels[0], 'vs', labels[1], comparison)
