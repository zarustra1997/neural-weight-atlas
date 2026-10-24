#!/usr/bin/env python3
"""Impressão digital LINEAR dos pesos: um "hash" com o qual dá para fazer contas (verificação de Freivalds, 1977).

Para cada matriz de especialista W (2880 × 2880) guarda-se f = rᵀW, com r aleatório e secreto (2880 números).
Quem afirma ter calculado y = W·x é conferido com r·y ≈ f·x: duas contas de 2880 em vez de 8,3 milhões.
- O SHA-256 prova "este é o arquivo"; f prova "esta conta usou estes pesos".
- f é linear: f(W1 + W2) = f(W1) + f(W2) e f(a·W) = a·f(W). O SHA não tem essa propriedade.
- Um peso trocado só passa despercebido se não alterar a saída (entrada zero naquela coluna) ou com
  probabilidade desprezível sobre r. Com r público, um adversário poderia forjar y; por isso r é secreto.
Mede nos 2.304 pesos de especialistas do gpt-oss original (blob só leitura, decodificação MXFP4 do
gptoss_forward.py validado). Só CPU, < 1 GB. Grava results/v4/fingerprint.json.
"""
from __future__ import annotations

import hashlib
import json
import time

import gptoss_forward as gf

np = gf.np
OUT = gf.ROOT / 'results' / 'v4' / 'fingerprint.json'
MATRICES = ('ffn_gate_exps.weight', 'ffn_up_exps.weight', 'ffn_down_exps.weight')
LAYERS = 24
EXPERTS = 32
SEED = 20261021              # no uso real, r é sorteado e mantido em segredo; aqui é fixo para reproduzir
CHECKS_PER_MATRIX = 4        # matrizes sorteadas recebem x novos a cada conferência
SAMPLED_MATRICES = 64
CALIBRATION_MATRICES = 16    # contas honestas só para medir o ruído do float32 (separadas do teste)
NOISE_MARGIN = 10            # limite = 10 × o maior resíduo honesto da calibração
SMALL_CHANGE_FRACTION = 0.1  # troca de um único peso por 10% do desvio típico dos pesos daquela matriz


def residual(r, y, f, x):
    """|r·y − f·x| relativo ao tamanho da conta; ~1e-7 numa conta honesta em float32."""
    lhs, rhs = float(np.dot(r, y.astype(np.float64))), float(np.dot(f, x.astype(np.float64)))
    scale = float(np.linalg.norm(r) * np.linalg.norm(y.astype(np.float64))) + 1e-30
    return abs(lhs - rhs) / scale


def build(weights, rng):
    r = rng.standard_normal(2880)
    prints, shas, seconds = {}, {}, 0.0
    for layer in range(LAYERS):
        for name in MATRICES:
            for e in range(EXPERTS):
                w = weights.tensor(f'blk.{layer}.{name}', e)
                t = time.perf_counter()
                prints[(layer, name, e)] = (r @ w.astype(np.float64))  # f = rᵀW (W: saída × entrada)
                seconds += time.perf_counter() - t
                shas[(layer, name, e)] = hashlib.sha256(w.tobytes()).hexdigest()
    return r, prints, shas, seconds


def calibrate(weights, r, prints, picks, rng):
    """Ruído de arredondamento de contas honestas em float32, em matrizes que não entram no teste."""
    noise = []
    for layer, name, e in picks:
        w = weights.tensor(f'blk.{layer}.{name}', e)
        for _ in range(CHECKS_PER_MATRIX):
            x = rng.standard_normal(w.shape[1]).astype(np.float32)
            noise.append(residual(r, w @ x, prints[(layer, name, e)], x))
    return NOISE_MARGIN * max(noise), max(noise)


def checks(weights, r, prints, rng):
    keys = list(prints)
    order = [keys[i] for i in rng.choice(len(keys), CALIBRATION_MATRICES + SAMPLED_MATRICES, replace=False)]
    TOLERANCE, noise_max = calibrate(weights, r, prints, order[:CALIBRATION_MATRICES], rng)
    picks = order[CALIBRATION_MATRICES:]
    honest, one_weight, small_weight, wrong_expert, linear = [], [], [], [], []
    matmul_s, verify_s = 0.0, 0.0
    for layer, name, e in picks:
        w = weights.tensor(f'blk.{layer}.{name}', e)
        f = prints[(layer, name, e)]
        std = float(w.std())
        other = prints[(layer, name, (e + 1) % EXPERTS)]
        for _ in range(CHECKS_PER_MATRIX):
            x = rng.standard_normal(w.shape[1]).astype(np.float32)
            t = time.perf_counter(); y = w @ x; matmul_s += time.perf_counter() - t
            t = time.perf_counter(); honest.append(residual(r, y, f, x)); verify_s += time.perf_counter() - t
            i, j = int(rng.integers(w.shape[0])), int(rng.integers(w.shape[1]))
            for delta, bucket in ((std, one_weight), (SMALL_CHANGE_FRACTION * std, small_weight)):
                bad = y.copy(); bad[i] += np.float32(delta) * x[j]  # o efeito exato de trocar só W[i, j]
                bucket.append(residual(r, bad, f, x))
            wrong_expert.append(residual(r, y, other, x))
        # linearidade: f(W + W2) = f(W) + f(W2)
        w2 = weights.tensor(f'blk.{layer}.{name}', (e + 1) % EXPERTS)
        summed = r @ (w.astype(np.float64) + w2.astype(np.float64))
        linear.append(float(np.max(np.abs(summed - (f + other))) / (np.max(np.abs(summed)) + 1e-30)))
    n = len(honest)
    rate = lambda xs: sum(v > TOLERANCE for v in xs) / len(xs)
    return {'conferencias': n, 'matrizes_sorteadas': SAMPLED_MATRICES, 'tolerancia_relativa': TOLERANCE,
            'ruido_honesto_max_calibracao': noise_max, 'margem_sobre_o_ruido': NOISE_MARGIN,
            'conta_honesta': {'aceitas': sum(v <= TOLERANCE for v in honest) / n, 'residuo_max': max(honest),
                              'residuo_mediano': float(np.median(honest))},
            'um_peso_trocado_1_desvio': {'detectadas': rate(one_weight), 'residuo_min': min(one_weight)},
            'um_peso_trocado_0_1_desvio': {'detectadas': rate(small_weight), 'residuo_min': min(small_weight)},
            'especialista_errado': {'detectadas': rate(wrong_expert), 'residuo_min': min(wrong_expert)},
            'linearidade_erro_relativo_max': max(linear),
            'tempo_por_multiplicacao_us': round(matmul_s / n * 1e6, 1),
            'tempo_por_conferencia_us': round(verify_s / n * 1e6, 1)}


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    started = time.perf_counter()
    weights = gf.Weights()
    try:
        r, prints, shas, build_s = build(weights, rng)
        result = checks(weights, r, prints, rng)
    finally:
        weights.close()
    matrices = len(prints)
    fp_bytes = matrices * 2880 * 4  # guardado em float32
    expert_bytes = matrices * 2880 * 2880 * 17 // 32  # MXFP4: 17 bytes por 32 pesos
    report = {'modelo': gf.SOURCE.name, 'matrizes': matrices, 'semente_fixa_para_reproduzir': SEED,
              'tamanho_impressoes_bytes': fp_bytes, 'tamanho_especialistas_mxfp4_bytes': expert_bytes,
              'fracao_do_tamanho': fp_bytes / expert_bytes, 'tempo_para_calcular_todas_s': round(build_s, 1),
              **result, 'sha256_das_matrizes_decodificadas_amostra': {f'{l}/{n.split("_")[1]}/{e}': s[:16]
                  for (l, n, e), s in list(shas.items())[:3]},
              'segundos_total': round(time.perf_counter() - started, 1), 'pico_rss_bytes': gf.rss_check()}
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps(report, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
