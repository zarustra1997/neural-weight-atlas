#!/usr/bin/env python3
"""Calibration tokens for counting gpt-oss expert usage (A2): the model's own conversations plus local prose.

generate: asks gpt-oss (Ollama) the prompts in results/experts/calibration_prompts_v1.json and keeps reasoning +
          answer. Prompts overlapping the 120 benchmark questions or the H01-H30 holdout are refused.
tokenize: rebuilds each conversation in the harmony chat format, adds local pt/en prose (code fences removed),
          tokenizes with the llama-server bundled in Ollama.app (CPU) and cuts sequences of <= 512 tokens.
Writes results/experts/calibration_*.json. The model blob is only read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
import unicodedata
import urllib.request

import bench120
import reference_topk as ref

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'experts'
PROMPTS = OUT / 'calibration_prompts_v1.json'
TRANSCRIPTS = OUT / 'calibration_transcripts_v1.json'
TOKENS = OUT / 'calibration_tokens_v1.json'
HOLDOUT = ROOT / 'results' / 'holdout' / 'holdout_H01_H30.json'
OLLAMA = 'http://127.0.0.1:11434'
MODEL = 'gpt-oss-20b-mini'
THINK = 'low'
TEMPERATURE = 0.7
SEED = 20261018
ANSWER_TOKENS = 700
SEQUENCE_TOKENS = 512
MIN_TOTAL_TOKENS = 30000
MAX_CODE_SHARE = 0.20
CODE_KINDS = {'codigo', 'code'}
SYSTEM = ('You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2024-06\n'
          'Current date: 2026-10-18\n\nReasoning: low\n\n# Valid channels: analysis, commentary, final. '
          'Channel must be included for every message.')
PROSE = {'pt': [ROOT / n for n in ('ATLAS_REPORT.md', 'METAL_REPORT.md', 'RUST_READER_REPORT.md', 'GPS_PIPELINE_REPORT.md',
                                   'MUSE_VIABILIDADE.md')] + [ROOT.parent / 'NEURAL_WEIGHT_ATLAS_V1' / 'REPORT.md'],
         'en': [ROOT.parent.parent / 'n8n-master' / 'CONTRIBUTING.md',
                ROOT.parent.parent / 'timesfm-master (terceiro, baixável) 094957' / 'README.md']}
PROSE_TOKENS_PER_LANGUAGE = 4000


def normalized(text):
    text = unicodedata.normalize('NFKD', text.lower())
    return re.sub(r'[^a-z0-9 ]+', ' ', ''.join(c for c in text if not unicodedata.combining(c))).split()


def forbidden_questions():
    qs = [q['question'] for q in json.loads((bench120.OUT / 'questions.json').read_text())]
    qs += [h['question'] for h in json.loads(HOLDOUT.read_text())] if HOLDOUT.exists() else []
    return [' '.join(normalized(q)) for q in qs]


def check_prompts(prompts):
    """Refuse a prompt that contains, or is contained in, a benchmark/holdout question."""
    bad = [p['text'] for p in prompts for q in forbidden_questions()
           if (n := ' '.join(normalized(p['text']))) and (q in n or n in q)]
    if bad:
        raise SystemExit(f'calibration prompts overlap the benchmark/holdout: {bad[:3]}')


def chat(prompt):
    body = {'model': MODEL, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False, 'think': THINK,
            'options': {'temperature': TEMPERATURE, 'seed': SEED, 'num_predict': ANSWER_TOKENS}}
    req = urllib.request.Request(OLLAMA + '/api/chat', json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=900) as r:
        m = json.loads(r.read())['message']
    return m.get('thinking', ''), m.get('content', '')


def generate():
    prompts = json.loads(PROMPTS.read_text())['prompts']
    check_prompts(prompts)
    done = json.loads(TRANSCRIPTS.read_text()) if TRANSCRIPTS.exists() else {'model': MODEL, 'items': []}
    seen = {i['text'] for i in done['items']}
    started = time.perf_counter()
    for p in prompts:
        if p['text'] in seen:
            continue
        thinking, answer = chat(p['text'])
        done['items'].append({**p, 'thinking': thinking, 'answer': answer})
        TRANSCRIPTS.write_text(json.dumps(done, ensure_ascii=False, indent=1) + '\n')  # resumable
        print(len(done['items']), p['lang'], p['kind'], len(thinking) + len(answer), 'chars', flush=True)
    subprocess.run(['ollama', 'stop', MODEL], capture_output=True)
    print('generate seconds', round(time.perf_counter() - started))


def harmony(item):
    return (f"<|start|>system<|message|>{SYSTEM}<|end|><|start|>user<|message|>{item['text']}<|end|>"
            f"<|start|>assistant<|channel|>analysis<|message|>{item['thinking']}<|end|>"
            f"<|start|>assistant<|channel|>final<|message|>{item['answer']}<|return|>")


def tokenize_special(text):
    return ref.post('/tokenize', {'content': text, 'add_special': False, 'parse_special': True})['tokens']


def chunks(tokens):
    return [tokens[i:i + SEQUENCE_TOKENS] for i in range(0, len(tokens), SEQUENCE_TOKENS) if len(tokens[i:i + SEQUENCE_TOKENS]) > 8]


def tokenize():
    items = json.loads(TRANSCRIPTS.read_text())['items']
    texts, sequences, sources = [], [], []
    process = subprocess.Popen(ref.command(), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    guard = ref.Guard(process)
    guard.start()
    try:
        ref.wait_ready(process)
        for item in items:
            text = harmony(item)
            texts.append(text)
            for s in chunks(tokenize_special(text)):
                sequences.append(s)
                sources.append({'source': 'conversation', 'lang': item['lang'], 'code': item['kind'] in CODE_KINDS})
        for lang, paths in PROSE.items():
            budget = PROSE_TOKENS_PER_LANGUAGE
            for path in paths:
                if budget <= 0 or not path.exists():
                    continue
                text = ref.plain_text(path)
                tokens = ref.tokenize(text)[:budget]
                texts.append(text)
                budget -= len(tokens)
                for s in chunks(tokens):
                    sequences.append(s)
                    sources.append({'source': path.name, 'lang': lang, 'code': False})
    finally:
        guard.done.set()
        process.terminate()
        process.wait(timeout=30)
    total = sum(map(len, sequences))
    code = sum(len(s) for s, src in zip(sequences, sources) if src['code']) / total
    by_lang = {l: sum(len(s) for s, src in zip(sequences, sources) if src['lang'] == l) for l in ('pt', 'en')}
    result = {'model_sha256': ref.DIGEST, 'text_sha256': hashlib.sha256('\n\x00\n'.join(texts).encode()).hexdigest(),
              'tokens': total, 'sequences': len(sequences), 'code_share': code, 'tokens_by_language': by_lang,
              'meets_minimum': total >= MIN_TOTAL_TOKENS and code <= MAX_CODE_SHARE,
              'guard_stop': guard.reason, 'sources': sources, 'token_ids': sequences}
    TOKENS.write_text(json.dumps(result) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('sources', 'token_ids')}, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('step', choices=('check', 'generate', 'tokenize'))
    step = p.parse_args().step
    if step == 'check':
        check_prompts(json.loads(PROMPTS.read_text())['prompts'])
        print('prompts ok: no overlap with the 120 questions or the holdout')
    else:
        {'generate': generate, 'tokenize': tokenize}[step]()
