#!/usr/bin/env python3
"""Reference next-token top-5 for the ORIGINAL gpt-oss blob, from the llama-server bundled in Ollama.app (CPU only).

Writes the format gptoss_forward.py validate() expects: passed, model_sha256_from_blob_name and cases with name,
token ids and top5 [{id, logprob}] (>=5 cases, one with exactly 512 tokens). Raw text, no chat template.
Memory guard: stop if swap grows or system free memory drops (RSS is not used: on macOS it counts the mmap'd
file pages, which the system can drop at any time). The blob is only read. Writes results/experts/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess
import threading
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'experts'
SERVER = Path('/Applications/Ollama.app/Contents/Resources/llama-server')
DIGEST = '8ecb5d4ad6172c63f9cbf1a07a04e525985a8a11a6e516da02b674c4010de716'
SOURCE = Path.home() / '.ollama/models/blobs' / f'sha256-{DIGEST}'
PORT = 8794
URL = f'http://127.0.0.1:{PORT}'
THREADS = 4
TOP_K = 5
LONG_TOKENS = 512
MEDIUM_TOKENS = 128
READY_TIMEOUT_SECONDS = 300
REQUEST_TIMEOUT_SECONDS = 600
GUARD_POLL_SECONDS = 0.5
MAX_SWAP_GROWTH_BYTES = 1_500_000_000
MIN_FREE_PERCENT = 10
SHORT_CASES = {
    'pt_curto': 'A fotossíntese é o processo pelo qual as plantas',
    'en_curto': 'The capital of France is',
    'codigo': 'def fibonacci(n):\n    if n < 2:\n        return n\n    return',
    'conta': 'Quanto é 7 vezes 8? A resposta é',
}
LONG_SOURCE = ROOT / 'REDUCE_REPORT.md'
MEDIUM_SOURCE = ROOT.parent.parent / 'n8n-master' / 'CONTRIBUTING.md'


def command():
    return [str(SERVER), '-m', str(SOURCE), '-ngl', '0', '--no-repack', '--no-warmup', '-c', '1024', '-b', '512',
            '-ub', '128', '-np', '1', '-t', str(THREADS), '-ctk', 'f32', '-ctv', 'f32', '-fa', 'off',
            '--host', '127.0.0.1', '--port', str(PORT)]


def swap_used_bytes():
    text = subprocess.run(['sysctl', '-n', 'vm.swapusage'], capture_output=True, text=True).stdout
    value, unit = re.search(r'used = ([\d.]+)([MG])', text).groups()
    return int(float(value) * (1 << (20 if unit == 'M' else 30)))


def free_percent():
    return int(subprocess.run(['sysctl', '-n', 'kern.memorystatus_level'], capture_output=True, text=True).stdout)


class Guard(threading.Thread):
    """Kills the server when the Mac starts swapping or runs out of free memory."""

    def __init__(self, process):
        super().__init__(daemon=True)
        self.process, self.reason, self.done = process, None, threading.Event()
        self.swap_start, self.swap_peak, self.free_min = swap_used_bytes(), 0, 100

    def run(self):
        while not self.done.is_set() and self.process.poll() is None:
            growth, free = swap_used_bytes() - self.swap_start, free_percent()
            self.swap_peak, self.free_min = max(self.swap_peak, growth), min(self.free_min, free)
            if growth > MAX_SWAP_GROWTH_BYTES or free < MIN_FREE_PERCENT:
                self.reason = {'swap_growth_bytes': growth, 'free_percent': free}
                self.process.terminate()
            time.sleep(GUARD_POLL_SECONDS)


def post(path, body):
    req = urllib.request.Request(URL + path, json.dumps(body).encode(), {'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=REQUEST_TIMEOUT_SECONDS) as r:
        return json.loads(r.read())


def wait_ready(process):
    deadline = time.time() + READY_TIMEOUT_SECONDS
    while time.time() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f'server exited before readiness: {process.returncode}')
        try:
            with urllib.request.urlopen(URL + '/health', timeout=5) as r:
                if json.loads(r.read()).get('status') == 'ok':
                    return
        except OSError:
            pass
        time.sleep(1)
    raise TimeoutError('server not ready')


def tokenize(text):
    return post('/tokenize', {'content': text, 'add_special': False})['tokens']


def plain_text(path):
    """Prose only: drop fenced code blocks."""
    return re.sub(r'```.*?```', '', Path(path).read_text(errors='replace'), flags=re.S)


def top5(tokens):
    d = post('/completion', {'prompt': tokens, 'n_predict': 1, 'n_probs': TOP_K, 'temperature': 0,
                             'cache_prompt': False, 'post_sampling_probs': False})
    first = d['completion_probabilities'][0]
    return [{'id': p['id'], 'token': p.get('token'), 'logprob': p['logprob']} for p in first['top_logprobs'][:TOP_K]]


def cases():
    out = [{'name': name, 'tokens': tokenize(text)} for name, text in SHORT_CASES.items()]
    out.append({'name': 'pt_longo_512', 'source': LONG_SOURCE.name, 'tokens': tokenize(plain_text(LONG_SOURCE))[:LONG_TOKENS]})
    out.append({'name': 'en_medio_128', 'source': MEDIUM_SOURCE.name, 'tokens': tokenize(plain_text(MEDIUM_SOURCE))[:MEDIUM_TOKENS]})
    return out


def run(output):
    OUT.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    before = SOURCE.stat()
    started = time.perf_counter()
    log = (OUT / (output.stem + '.log')).open('w')
    process = subprocess.Popen(command(), stdout=log, stderr=subprocess.STDOUT)
    guard = Guard(process)
    guard.start()
    result = {'status': 'failed', 'passed': False, 'source': str(SOURCE), 'model_sha256_from_blob_name': DIGEST,
              'command': command(), 'cases': []}
    try:
        wait_ready(process)
        result['cases'] = cases()
        for c in result['cases']:
            c['top5'] = top5(c['tokens'])
        lengths = [len(c['tokens']) for c in result['cases']]
        result['passed'] = (len(result['cases']) >= 5 and LONG_TOKENS in lengths
                            and all(len(c['top5']) == TOP_K for c in result['cases']))
        result['status'] = 'ok' if result['passed'] else 'incomplete'
    except Exception as e:  # keep the failure on disk, never a partial "passed"
        result['error'] = f'{type(e).__name__}: {e}'
    finally:
        guard.done.set()
        process.terminate()
        process.wait(timeout=30)
        log.close()
    after = SOURCE.stat()
    result.update({'guard_stop': guard.reason, 'swap_growth_peak_bytes': guard.swap_peak,
                   'free_percent_min': guard.free_min, 'seconds': round(time.perf_counter() - started, 1),
                   'source_stat_unchanged': (before.st_ino, before.st_size, before.st_mtime_ns) ==
                                            (after.st_ino, after.st_size, after.st_mtime_ns)})
    output.write_text(json.dumps(result, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('cases', 'command')}, ensure_ascii=False, indent=1))
    for c in result['cases']:
        print(c['name'], len(c['tokens']), [(p['token'], round(p['logprob'], 3)) for p in c.get('top5', [])])
    return result['passed']


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, default=OUT / 'forward_reference_20261018.json')
    raise SystemExit(0 if run(p.parse_args().output) else 1)
