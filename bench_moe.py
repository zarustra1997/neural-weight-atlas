#!/usr/bin/env python3
"""GPT-OSS 20B (MoE, a small Muse) with the 120 questions of bench120.py: quality, memory, SSD reads and speed.

Targets: the model inside Ollama, or a llama-server already started (e.g. with --cpu-moe, experts read from the
file through mmap). Same questions and strict scoring as bench120.py (imported, not edited). Writes only results/moe/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import subprocess
import time
import urllib.request

import bench120

OUT = bench120.ROOT / 'results' / 'moe'
OLLAMA = bench120.OLLAMA
SEED = bench120.SEED
CONTEXT = bench120.CONTEXT
ANSWER_TOKENS = 1024          # reasoning model: thinking tokens come before the answer
REASONING = 'low'
SPEED_TOKENS = 256
SPEED_REPEATS = 3
PAGE_BYTES = 16384            # macOS arm64 page size (vm_stat)
TIMEOUT_SECONDS = 900


def pageins():
    out = subprocess.run(['vm_stat'], capture_output=True, text=True).stdout
    line = next(l for l in out.splitlines() if l.startswith('Pageins'))
    return int(line.split(':')[1].strip().rstrip('.'))


def rss_bytes(pid):
    out = subprocess.run(['ps', '-o', 'rss=', '-p', str(pid)], capture_output=True, text=True).stdout.strip()
    return int(out) * 1024 if out else None


def post(url, body):
    req = urllib.request.Request(url, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
        data = json.loads(r.read())
    return data, time.perf_counter() - start


def ask_ollama(model, prompt, tokens, think=REASONING, num_gpu=None):
    d, wall = post(OLLAMA + '/api/chat', {'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False,
                                          'think': think,
                                          'options': {'temperature': 0, 'seed': SEED, 'num_predict': tokens, 'num_ctx': CONTEXT,
                                                      **({'num_gpu': num_gpu} if num_gpu is not None else {})}})
    m = d['message']
    return {'answer': m.get('content', ''), 'thinking': m.get('thinking', ''), 'generated': d.get('eval_count'),
            'tokens_per_second': d['eval_count'] / d['eval_duration'] * 1e9 if d.get('eval_duration') else None,
            'wall_seconds': round(wall, 3)}


def ask_server(url, prompt, tokens):
    d, wall = post(url + '/v1/chat/completions', {'messages': [{'role': 'user', 'content': prompt}], 'temperature': 0,
                                                 'seed': SEED, 'max_tokens': tokens,
                                                 'chat_template_kwargs': {'reasoning_effort': REASONING}})
    m, t = d['choices'][0]['message'], d.get('timings', {})
    return {'answer': m.get('content') or '', 'thinking': m.get('reasoning_content') or '', 'generated': t.get('predicted_n'),
            'tokens_per_second': t.get('predicted_per_second'), 'wall_seconds': round(wall, 3)}


def run(args):
    OUT.mkdir(parents=True, exist_ok=True)
    think = {'false': False, 'true': True}.get(args.think, args.think)
    ask = (lambda p, n: ask_ollama(args.model, p, n, think, args.num_gpu)) if args.target == 'ollama' else (lambda p, n: ask_server(args.url, p, n))
    questions = json.loads((bench120.OUT / 'questions.json').read_text())
    ask('Responda só: ok', 16)  # load / warm-up, not scored
    loaded = subprocess.run(['ollama', 'ps'], capture_output=True, text=True).stdout if args.target == 'ollama' else None
    before, rss_peak, started = pageins(), 0, time.perf_counter()
    rows = []
    for q in ([] if args.speed_only else questions):
        r = ask(q['question'], ANSWER_TOKENS)
        rows.append({'id': q['id'], 'category': q['category'], **r,
                     'exact': bench120.exact(r['answer'], q['accept'], q['case_sensitive']),
                     'word_rule': bench120.word_in_first_line(r['answer'], q['accept']),
                     'empty_answer': not r['answer'].strip()})
        if args.pid:
            rss_peak = max(rss_peak, rss_bytes(args.pid) or 0)
    questions_seconds = time.perf_counter() - started
    read_q = (pageins() - before) * PAGE_BYTES
    speed, before = [], pageins()
    for _ in range(SPEED_REPEATS):
        for prompt in bench120.SPEED_PROMPTS:
            s = ask(prompt, SPEED_TOKENS)
            speed.append({'generated': s['generated'], 'tokens_per_second': s['tokens_per_second']})
            if args.pid:
                rss_peak = max(rss_peak, rss_bytes(args.pid) or 0)
    read_s = (pageins() - before) * PAGE_BYTES
    gen_s = sum(s['generated'] or 0 for s in speed)
    if args.speed_only:
        summary = {'label': args.label, 'model': args.model, 'num_gpu': args.num_gpu, 'ollama_ps': loaded,
                   'speed_tokens_per_second_median': statistics.median(s['tokens_per_second'] for s in speed if s['tokens_per_second']),
                   'speed_samples': speed, 'ssd_bytes_read_during_speed': read_s,
                   'ssd_bytes_read_per_generated_token_speed': read_s / gen_s if gen_s else None,
                   'note': 'speed and memory only, no questions; vm_stat pageins are system-wide'}
        (OUT / f'speed_{args.label}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
        print(json.dumps({x: v for x, v in summary.items() if x != 'speed_samples'}, ensure_ascii=False, indent=1))
        return
    base = json.loads((bench120.OUT / f"answers_{bench120.BASELINE.replace(':', '_')}.json").read_text())
    k = sum(r['exact'] for r in rows)
    summary = {'label': args.label, 'target': args.target, 'model': args.model, 'url': args.url, 'ollama_ps': loaded,
               'settings': {'temperature': 0, 'seed': SEED, 'answer_tokens': ANSWER_TOKENS, 'think': args.think, 'num_gpu': args.num_gpu,
                            'num_ctx': CONTEXT, 'speed_tokens': SPEED_TOKENS, 'speed_repeats': SPEED_REPEATS},
               'exact': k, 'exact_wilson_95': bench120.wilson(k, len(rows)), 'word_rule': sum(r['word_rule'] for r in rows),
               'empty_answers': sum(r['empty_answer'] for r in rows),
               'vs_qwen_original': bench120.mcnemar([b['exact'] for b in base], [r['exact'] for r in rows]),
               'by_category': {c: f"{sum(r['exact'] for r in rows if r['category'] == c)}/{sum(r['category'] == c for r in rows)}"
                               for c in dict.fromkeys(r['category'] for r in rows)},
               'questions_seconds': round(questions_seconds, 1),
               'speed_tokens_per_second_median': statistics.median(s['tokens_per_second'] for s in speed if s['tokens_per_second']),
               'speed_samples': speed,
               'ssd_bytes_read_during_questions': read_q, 'ssd_bytes_read_during_speed': read_s,
               'ssd_bytes_read_per_generated_token_speed': read_s / gen_s if gen_s else None,
               'process_rss_peak_bytes': rss_peak or None,
               'note': 'vm_stat pageins are system-wide (other apps included); cached file pages are not re-read'}
    (OUT / f'answers_{args.label}.json').write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n')
    (OUT / f'summary_{args.label}.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({x: v for x, v in summary.items() if x not in ('speed_samples',)}, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--target', choices=('ollama', 'server'), required=True)
    p.add_argument('--model', default='gpt-oss:20b')
    p.add_argument('--url', default='http://127.0.0.1:8793')
    p.add_argument('--label', required=True)
    p.add_argument('--pid', type=int, help='process to sample RSS from (llama-server)')
    p.add_argument('--num-gpu', type=int, help='Ollama num_gpu (layers on the GPU); 0 = CPU only, weights read from the file')
    p.add_argument('--speed-only', action='store_true', help='skip the 120 questions: memory (ollama ps) and speed only')
    p.add_argument('--think', default=REASONING, help="Ollama think: low/medium/high for gpt-oss, true/false for qwen3/deepseek-r1")
    run(p.parse_args())
