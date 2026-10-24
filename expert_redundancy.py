#!/usr/bin/env python3
"""P4 (NWA_V4_PROVAS.md): do the 32 experts of a gpt-oss layer share structure?

For each sampled layer and matrix (gate, up, down), the 32 experts are decoded exactly from the original MXFP4 blob
(gptoss_forward.decode_mxfp4, validated) and compared:
  1. energy explained by the mean expert;
  2. energy a SHARED rank-r input subspace captures (eigenvectors of sum_i W_i^T W_i), against each expert's OWN
     rank-r subspace and against the isotropic line r/2880;
  3. relative RMS error of W_i ~ A_i B (B shared, r x 2880) at the K=20 memory budget, r = 1745.
Read-only, RAM < 2 GB. Writes results/v4/expert_redundancy.json.
"""
from __future__ import annotations

import json
from pathlib import Path
import time

import gptoss_forward as gf

np = gf.np
OUT = gf.ROOT / 'results' / 'v4'
LAYERS = (0, 3, 6, 9, 12, 15, 18, 23)
MATRICES = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')
RANKS = (360, 720, 1440, 1745, 2160)
BUDGET_RANK = 1745  # 33 * 2880 * r <= 20 * 2880^2: 32 experts as A_i plus one shared B fit in the K=20 memory
SUCCESS_MARGIN, ABANDON_MARGIN, MAX_RMS = 0.15, 0.05, 0.10
OWN_SPECTRUM_EXPERTS = range(0, 32, 4)  # 8 of 32 experts for the per-expert spectrum (time budget)


def expert_matrices(weights, layer, name):
    experts = weights.tensors[f'blk.{layer}.{name}']['dims'][-1]
    return [weights.tensor(f'blk.{layer}.{name}', e) for e in range(experts)]  # float32, exact MXFP4 values


def captured(gram, total, basis_rank, vectors):
    """Energy fraction a set of input-space directions captures: trace(V^T G V) / trace(G)."""
    v = vectors[:, :basis_rank]
    return float(np.einsum('ij,ij->', v, gram @ v) / total)


def analyse(ws):
    n = len(ws)
    dim = ws[0].shape[1]
    energy = np.array([np.sum(w.astype(np.float64) ** 2) for w in ws])
    mean = sum(ws) / n
    mean_share = 1 - sum(np.sum((w - mean).astype(np.float64) ** 2) for w in ws) / energy.sum()
    gram = np.zeros((dim, dim))
    for w in ws:
        gram += (w.T @ w).astype(np.float64)
    evals, evecs = np.linalg.eigh(gram)
    shared = evecs[:, ::-1]
    own = {}
    for w in (ws[i] for i in OWN_SPECTRUM_EXPERTS):
        s = np.linalg.eigvalsh((w.T @ w).astype(np.float64))[::-1]
        cum = np.cumsum(s) / s.sum()
        for r in RANKS:
            own.setdefault(r, []).append(float(cum[r - 1]))
    total = float(np.trace(gram))
    rows = {}
    for r in RANKS:
        shared_cap = captured(gram, total, r, shared)
        rows[str(r)] = {'shared_subspace': shared_cap, 'own_subspace_mean': float(np.mean(own[r])),
                        'isotropic_line': r / dim, 'shared_minus_isotropic': shared_cap - r / dim,
                        'relative_rms_error_shared': float(np.sqrt(max(0.0, 1 - shared_cap)))}
    return {'mean_expert_energy_share': float(mean_share), 'by_rank': rows}


def verdict(results):
    budget = [m['by_rank'][str(BUDGET_RANK)] for m in results]
    margin = min(b['shared_minus_isotropic'] for b in budget)
    rms = max(b['relative_rms_error_shared'] for b in budget)
    mean_share = max(m['mean_expert_energy_share'] for m in results)
    if margin >= SUCCESS_MARGIN and rms <= MAX_RMS:
        return 'SUCESSO', margin, rms, mean_share
    if mean_share < 0.05 and margin <= ABANDON_MARGIN:
        return 'ABANDONO', margin, rms, mean_share
    return 'INCONCLUSIVO', margin, rms, mean_share


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    weights = gf.Weights()
    results = []
    try:
        for layer in LAYERS:
            for name in MATRICES:
                t = time.perf_counter()
                row = {'layer': layer, 'matrix': name, **analyse(expert_matrices(weights, layer, name))}
                row['seconds'] = round(time.perf_counter() - t, 1)
                results.append(row)
                b = row['by_rank'][str(BUDGET_RANK)]
                print(json.dumps({'layer': layer, 'matrix': name.split('.')[0], 'mean_share': round(row['mean_expert_energy_share'], 4),
                                  'shared_1745': round(b['shared_subspace'], 4), 'own_1745': round(b['own_subspace_mean'], 4),
                                  'iso_1745': round(b['isotropic_line'], 4), 'rms_1745': round(b['relative_rms_error_shared'], 4),
                                  's': row['seconds']}), flush=True)
    finally:
        weights.close()
    status, margin, rms, mean_share = verdict(results)
    report = {'preregistration': 'NWA_V4_PROVAS.md P4', 'model_sha256': gf.SOURCE.name.removeprefix('sha256-'),
              'layers': LAYERS, 'budget_rank': BUDGET_RANK, 'verdict': status,
              'worst_shared_minus_isotropic_at_budget': margin, 'worst_relative_rms_at_budget': rms,
              'best_mean_expert_energy_share': mean_share, 'results': results,
              'seconds': round(time.perf_counter() - started, 1), 'peak_rss_bytes': gf.rss_check()}
    (OUT / 'expert_redundancy.json').write_text(json.dumps(report, indent=1) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'results'}, indent=1))


if __name__ == '__main__':
    run()
