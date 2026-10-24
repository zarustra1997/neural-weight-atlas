#!/usr/bin/env python3
"""Rebuild and run the pruned gpt-oss-20b on the best device this computer has.

    python3 nwa.py          # build: pull gpt-oss:20b, cut to 20 experts per layer, add gpt-oss-20b-nwa to Ollama
    python3 nwa.py test     # the 120 questions + the 30 holdout questions on gpt-oss-20b-nwa
    python3 nwa.py chat     # talk to gpt-oss-20b-nwa

Device: a Mac runs Ollama directly (Metal GPU; Docker cannot reach it). Elsewhere it runs in Docker (compose.yaml):
NVIDIA GPU if nvidia-smi sees one, AMD GPU if /dev/kfd exists, otherwise the CPU. Python 3 standard library only.
"""
from __future__ import annotations

import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

import reduce_installed

ROOT = Path(__file__).resolve().parent
SOURCE_TAG = 'gpt-oss:20b'
MODEL = 'gpt-oss-20b-nwa'
LABEL = 'nwa'  # nome próprio: não sobrescreve os resultados publicados em results/prune
EXPERTS_KEPT = 20
OLLAMA_URL = 'http://127.0.0.1:11434'
OLLAMA_START_SECONDS = 60


def best_device():
    if platform.system() == 'Darwin':
        return 'mac'
    if shutil.which('nvidia-smi') and subprocess.run(['nvidia-smi', '-L'], capture_output=True).returncode == 0:
        return 'nvidia'
    if Path('/dev/kfd').exists():
        return 'amd'
    return 'cpu'


def ollama_running():
    try:
        with urllib.request.urlopen(OLLAMA_URL + '/api/version', timeout=2):
            return True
    except OSError:
        return False


def ensure_ollama():
    if not shutil.which('ollama'):
        raise SystemExit('Ollama not found; install it from https://ollama.com/download')
    if ollama_running():
        return
    subprocess.Popen(['ollama', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)  # no Docker, sem o app aberto
    deadline = time.monotonic() + OLLAMA_START_SECONDS
    while not ollama_running():
        if time.monotonic() > deadline:
            raise SystemExit('ollama serve did not start')
        time.sleep(1)


def step(*args):
    print('$', ' '.join(args), flush=True)
    if subprocess.run(args, cwd=ROOT).returncode:
        raise SystemExit(f'step failed: {" ".join(args)}')


def build():
    ensure_ollama()
    if not reduce_installed.QUANTIZE.exists():
        raise SystemExit(f'llama-quantize not found at {reduce_installed.QUANTIZE}; set LLAMA_QUANTIZE')
    step('ollama', 'pull', SOURCE_TAG)
    step(sys.executable, '-B', 'prune_experts.py', '--label', LABEL, '--usage', 'results/experts/usage.json',
         '--k', str(EXPERTS_KEPT))
    step(sys.executable, '-B', 'build_final.py', '--label', LABEL, '--ollama-name', MODEL)
    print(f'\nready: python3 nwa.py chat  (or: ollama run {MODEL})\n'
          'the model is now inside Ollama; the two GGUF files in results/prune/models can be deleted')


def test():
    ensure_ollama()
    step(sys.executable, '-B', 'bench_moe.py', '--target', 'ollama', '--model', MODEL, '--label', MODEL)
    step(sys.executable, '-B', 'bench_holdout.py', '--model', f'{MODEL}={MODEL}')


def chat():
    ensure_ollama()
    subprocess.run(['ollama', 'run', MODEL])


COMMANDS = {'build': build, 'test': test, 'chat': chat}


def in_docker(device, command):
    if not shutil.which('docker'):
        raise SystemExit('Docker not found; install Docker (with GPU support for NVIDIA/AMD) and run again')
    service = f'nwa-{device}'
    print(f'device: {device} -> docker compose service {service}', flush=True)
    sys.exit(subprocess.run(['docker', 'compose', '--profile', device, 'run', '--rm', service, command], cwd=ROOT).returncode)


def main():
    command = sys.argv[1] if len(sys.argv) > 1 else 'build'
    if command not in COMMANDS:
        raise SystemExit(__doc__)
    if os.environ.get('NWA_IN_DOCKER'):
        return COMMANDS[command]()
    device = best_device()
    if device != 'mac':
        return in_docker(device, command)
    print('device: mac -> Ollama directly (Metal GPU)', flush=True)
    COMMANDS[command]()


if __name__ == '__main__':
    main()
