#!/usr/bin/env python3
"""P3 (NWA_V4_PROVAS.md): the SSD ceiling for streaming Muse experts on this Mac, measured.

Muse Spark Open (model page): 62 layers, 256 experts, 8 active per token. One routed expert ~50M parameters
(ESTIMATED from the page's numbers; config.json not read), ~9.86 MB at ~1.5625 bit (IQ1-class). With the dense
part resident in RAM, every token must read 8 x 62 = 496 such chunks if nothing is cached.
This reads chunks of that size at random 16 KB-aligned offsets of large files already on disk (Ollama blobs),
with the file cache off (F_NOCACHE), at 1, 4 and 8 reads in flight. Read-only. Writes results/v4/muse_io_limit.json.
"""
from __future__ import annotations

import concurrent.futures
import fcntl
import json
import os
from pathlib import Path
import random
import time

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'v4'
BLOBS = Path.home() / '.ollama' / 'models' / 'blobs'
F_NOCACHE = 48  # macOS fcntl: do not keep what is read in the file cache
PAGE = 16384
MUSE_LAYERS, MUSE_ACTIVE = 62, 8
EXPERT_PARAMS = 50_500_000  # ESTIMATED: (816B - attention - embeddings - vision) / (62 x 256)
BITS_PER_PARAM = 1.5625     # IQ1-class average
CHUNK = (int(EXPERT_PARAMS * BITS_PER_PARAM / 8) // PAGE) * PAGE
READS_PER_TOKEN = MUSE_LAYERS * MUSE_ACTIVE
TOKENS_PER_LEVEL = 4
QUEUE_DEPTHS = (1, 4, 8)
MIN_FILE_BYTES = 4 << 30
SKIP_RECENT = {'sha256-8ecb5d4ad6172c63f9cbf1a07a04e525985a8a11a6e516da02b674c4010de716'}  # read by P4 just before
SEED = 20261020


def sources():
    files = [p for p in BLOBS.iterdir() if p.stat().st_size >= MIN_FILE_BYTES and p.name not in SKIP_RECENT]
    if not files:
        raise SystemExit('no large files to read')
    return files


def open_uncached(path):
    fd = os.open(path, os.O_RDONLY)
    fcntl.fcntl(fd, F_NOCACHE, 1)
    return fd


def plan(files, count, rng):
    sizes = [f.stat().st_size for f in files]
    picks = []
    for _ in range(count):
        i = rng.randrange(len(files))
        picks.append((i, rng.randrange(0, (sizes[i] - CHUNK) // PAGE) * PAGE))
    return picks


def read_one(fds, pick):
    i, offset = pick
    data = os.pread(fds[i], CHUNK, offset)
    if len(data) != CHUNK:
        raise EOFError('short read')
    return len(data)


def level(files, depth, rng):
    fds = [open_uncached(f) for f in files]
    try:
        picks = plan(files, READS_PER_TOKEN * TOKENS_PER_LEVEL, rng)
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(depth) as pool:
            total = sum(pool.map(lambda p: read_one(fds, p), picks))
        seconds = time.perf_counter() - start
    finally:
        for fd in fds:
            os.close(fd)
    gbps = total / seconds / 1e9
    return {'reads_in_flight': depth, 'bytes': total, 'seconds': round(seconds, 2), 'gb_per_s': round(gbps, 2),
            'seconds_per_token': round(seconds / TOKENS_PER_LEVEL, 2), 'tokens_per_s_no_cache': round(gbps / (CHUNK * READS_PER_TOKEN / 1e9), 2)}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    files = sources()
    rng = random.Random(SEED)
    rows = [level(files, d, rng) for d in QUEUE_DEPTHS]
    best = max(r['tokens_per_s_no_cache'] for r in rows)
    with_cache = {f'{int(h * 100)}%_cache_hits': round(best / (1 - h), 2) for h in (0.3, 0.5, 0.8)}
    report = {'preregistration': 'NWA_V4_PROVAS.md P3', 'chunk_bytes': CHUNK, 'reads_per_token': READS_PER_TOKEN,
              'gb_per_token_no_cache': round(CHUNK * READS_PER_TOKEN / 1e9, 2), 'files': len(files),
              'file_bytes_total': sum(f.stat().st_size for f in files), 'levels': rows,
              'best_tokens_per_s_no_cache': best, 'if_cache_hits': with_cache,
              'verdict': 'CONFIRMADO' if 0.5 <= best <= 2.5 else ('REVISAR' if best > 5 else 'FORA_DA_FAIXA'),
              'note': 'routed experts only, dense part assumed resident; compute and dequantization not included (upper bound)'}
    (OUT / 'muse_io_limit.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps(report, indent=1))


if __name__ == '__main__':
    run()
