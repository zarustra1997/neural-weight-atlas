# Neural Weight Atlas — gpt-oss-20b num Mac de 16 GB

*English below.*

Corte guiado por uso dos especialistas (MoE) do `gpt-oss-20b`: de 32 para 20 especialistas por camada, escolhidos pelo
quanto o roteador realmente os usa. O modelo cai de 13,79 GB para 8,29 GB, passa a caber inteiro na GPU de um
MacBook Air M5 com 16 GB e fica 1,6× mais rápido, com o mesmo placar nos testes daqui.

Este repositório traz a receita, o código e os resultados brutos. **Não traz pesos**: o modelo cortado se refaz a partir
do `gpt-oss:20b` oficial com os scripts abaixo.

## Resultados

| Versão | Arquivo | Memória (`ollama ps`) | GPU | tok/s | 120 perguntas | 30 novas | Situação |
|---|---:|---:|---:|---:|---:|---:|---|
| `gpt-oss:20b` original | 13,79 GB | 13 GB | 82% (swap) | 18,1 | 117 | 30 | referência |
| **K=20 guiado** (`gpt-oss-20b-final`) | **8,29 GB** | **7,8–8,4 GB** | **100%** | **29,6** | **119** | **28** | **resultado** |
| K=16 guiado | 7,02 GB | 7,1 GB | 100% | 30,4 | 111 | — | reprovado |
| K=20 + atenção Q5_0, saída Q5_1, embedding IQ4_NL | 7,58 GB | 7,4 GB | 100% | 22,0 | 112 | 29 | reprovado |

- Nas 150 perguntas: original 147, K=20 guiado 147.
- O original lê ~6,1 MB do SSD por token gerado (não cabe na memória); o K=20 lê ~0,04 MB.
- As 30 perguntas novas (`results/holdout/`) não entraram na calibração; o SHA-256 delas está fixado em `bench_holdout.py`.

### Provas pré-registradas (`PRE_REGISTRO_V4.md`)

Critérios escritos antes de rodar; resultados negativos ficam registrados.

- **P1, fidelidade palavra a palavra — não atingiu o critério.** KL média de 0,284 nat e top-1 igual em 77,4% (91,3%
  só no texto que o próprio modelo gera). O critério era KL ≤ 0,05 e top-1 ≥ 95%. Ou seja: acerta o mesmo nestes
  testes, mas é um modelo diferente palavra a palavra.
- **P1, valor do mapa de uso — confirmado.** Com o mesmo tamanho, o corte guiado muda o modelo 3,2× menos que um corte
  aleatório de 20 e 2,1× menos que o K=16.
- **P3, teto de um MoE de 816B lido do SSD — confirmado.** 1,1–1,6 tok/s medidos, só pela leitura do disco.
- **P4, redundância entre especialistas — refutada para reconstrução exata.** A média explica 3–5% da energia; uma
  base compartilhada no orçamento do K=20 erra 0,32–0,54 (RMS relativo).
- **P2** dependia de um sistema privado e não foi executada.

### Impressão digital linear dos pesos (`weight_fingerprint.py`)

Para cada matriz W de especialista guarda-se f = rᵀW, com r aleatório e secreto (verificação de Freivalds). Quem afirma
ter calculado y = W·x é conferido com r·y ≈ f·x. Nas 2.304 matrizes do original: 26,5 MB de impressões, contas honestas
aceitas em 100%, especialista errado detectado em 100%, um único peso trocado detectado em 98,8% (1 desvio) e 84%
(0,1 desvio). Ao contrário do SHA-256, f é linear: f(W1 + W2) = f(W1) + f(W2).

## Limites

- Equivalência formal **não** demonstrada: sem margem de não inferioridade, e os testes estão perto do teto.
- A fidelidade (P1) foi medida no corte com atenção e saída em BF16, isolando o efeito do corte; não no arquivo final.
- Tudo foi medido em uma máquina: MacBook Air M5, 16 GB, macOS, Ollama.
- O forward NumPy (`gptoss_forward.py`) reproduz o top-1 do llama-server em 6/6 casos e 99,4% das escolhas do roteador,
  mas não passa na regra estrita de ±0,15 em 2 casos em português.
- `gptoss_forward.py` põe no `sys.path` a pasta `../NEURAL_WEIGHT_ATLAS_V1/vendor`, se existir (sobra da máquina
  original; ficou para não mudar o SHA-256). Sem ela, usa o `numpy` instalado.
- A calibração e a P1 usaram prosa local que não está aqui (`EXCLUIDO.md`). Para recalcular o mapa, troque os
  caminhos em `PROSE` por textos seus; o mapa publicado (`results/experts/usage.json`) já basta para refazer o corte.

## Como reproduzir

Requisitos: macOS com o app Ollama (o `llama-quantize` e o `llama-server` vêm dentro dele), Python 3 e `numpy`
(`requirements.txt`). Até ~50 GB livres no disco durante o processo (o intermediário pode ser apagado no fim).

```bash
ollama pull gpt-oss:20b                       # o blob deve ter SHA-256 8ecb5d4a…de716
python3 prune_experts.py --label k20-guiado --usage results/experts/usage.json --k 20 --ollama-name gpt-oss-20b-k20
python3 reduce_installed.py gpt-oss-20b-k20 --type MXFP4_MOE --name gpt-oss-20b-final
python3 bench_moe.py --target ollama --model gpt-oss-20b-final --label final
python3 bench_holdout.py --model gpt-oss-20b-final=final
```

- O corte confere o SHA-256 do original antes e depois (o original só é lido) e deve gerar o arquivo intermediário com
  SHA-256 `70df4e5c…317f5`.
- O `MXFP4_MOE` mantém os especialistas idênticos e põe o resto em Q8_0. Com dimensão 2880, quantizações K não servem.
- Mapa de uso do zero: `gptoss_calibration.py` → `gptoss_usage.py` (só CPU, menos de 2 GB de RAM).
- Provas: `gptoss_fidelity.py`, `expert_redundancy.py`, `muse_io_limit.py`, `weight_fingerprint.py`.

`gptoss_forward.py` está byte a byte igual ao usado nos resultados (o SHA-256 dele está nos JSON e `gptoss_usage.py`
o confere). Os SHA-256 de cada arquivo desta cópia estão em `MANIFEST.json`.

## Licença

Código e resultados: MIT (`LICENSE`). Os pesos do `gpt-oss-20b` são da OpenAI, sob Apache-2.0, e não são
redistribuídos aqui.

---

# Neural Weight Atlas — gpt-oss-20b on a 16 GB Mac

Usage-guided pruning of the `gpt-oss-20b` experts (MoE): from 32 to 20 experts per layer, chosen by how much the router
actually uses them. The model shrinks from 13.79 GB to 8.29 GB, fits entirely on the GPU of a 16 GB MacBook Air M5,
and runs 1.6× faster, with the same score on the tests here.

This repository holds the recipe, the code and the raw results. **It holds no weights**: the pruned model is rebuilt
from the official `gpt-oss:20b` with the scripts below.

## Results

| Version | File | Memory (`ollama ps`) | GPU | tok/s | 120 questions | 30 new | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `gpt-oss:20b` original | 13.79 GB | 13 GB | 82% (swap) | 18.1 | 117 | 30 | reference |
| **K=20 guided** (`gpt-oss-20b-final`) | **8.29 GB** | **7.8–8.4 GB** | **100%** | **29.6** | **119** | **28** | **result** |
| K=16 guided | 7.02 GB | 7.1 GB | 100% | 30.4 | 111 | — | rejected |
| K=20 + attention Q5_0, output Q5_1, embedding IQ4_NL | 7.58 GB | 7.4 GB | 100% | 22.0 | 112 | 29 | rejected |

- Over all 150 questions: original 147, K=20 guided 147.
- The original reads ~6.1 MB from the SSD per generated token (it does not fit in memory); K=20 reads ~0.04 MB.
- The 30 new questions (`results/holdout/`) were not used for calibration; their SHA-256 is pinned in `bench_holdout.py`.

### Pre-registered tests (`PRE_REGISTRO_V4.md`, in Portuguese)

Criteria were written before running; negative results are kept.

- **P1, token-level fidelity — criterion not met.** Mean KL 0.284 nat, top-1 agreement 77.4% (91.3% on text the model
  generates itself), against a criterion of KL ≤ 0.05 and top-1 ≥ 95%. Same answers on these tests, but a different
  model token by token.
- **P1, value of the usage map — confirmed.** At equal size, the guided cut changes the model 3.2× less than a random
  20-expert cut and 2.1× less than K=16.
- **P3, ceiling of an 816B MoE streamed from the SSD — confirmed.** 1.1–1.6 tok/s measured, from disk reads alone.
- **P4, expert redundancy — refuted for exact reconstruction.** The mean explains 3–5% of the energy; a shared basis at
  the K=20 memory budget has 0.32–0.54 relative RMS error.
- **P2** depended on a private system and was not run.

### Linear weight fingerprint (`weight_fingerprint.py`)

For each expert matrix W we keep f = rᵀW with a random, secret r (Freivalds' check). A claimed y = W·x is checked with
r·y ≈ f·x. On the original's 2,304 matrices: 26.5 MB of fingerprints, honest products accepted 100%, wrong expert
detected 100%, a single changed weight detected 98.8% (1 std) and 84% (0.1 std). Unlike SHA-256, f is linear:
f(W1 + W2) = f(W1) + f(W2).

## Limits

- Formal equivalence is **not** shown: no non-inferiority margin, and the tests are near ceiling.
- Fidelity (P1) was measured on the cut with BF16 attention and output, isolating the pruning; not on the final file.
- Everything was measured on one machine: MacBook Air M5, 16 GB, macOS, Ollama.
- The NumPy forward (`gptoss_forward.py`) matches llama-server's top-1 in 6/6 cases and 99.4% of router choices, but
  fails the strict ±0.15 rule in 2 Portuguese cases.
- `gptoss_forward.py` adds `../NEURAL_WEIGHT_ATLAS_V1/vendor` to `sys.path` if it exists (left over from the original
  machine; kept so its SHA-256 does not change). Without it, the installed `numpy` is used.
- Calibration and P1 used local prose that is not included (`EXCLUIDO.md`). To recompute the map, point `PROSE` to
  your own texts; the published map (`results/experts/usage.json`) is enough to redo the cut.

## Reproduce

Needs macOS with the Ollama app (it bundles `llama-quantize` and `llama-server`), Python 3 and `numpy`
(`requirements.txt`), and up to ~50 GB of free disk during the process (the intermediate file can be deleted at the end). Commands: see the block above. The pruning checks the original's
SHA-256 before and after (the original is only read) and should produce an intermediate file with SHA-256
`70df4e5c…317f5`. `MXFP4_MOE` keeps the experts identical and puts the rest in Q8_0; with dimension 2880, K-quants do
not apply.

`gptoss_forward.py` is byte-identical to the one used for the results (its SHA-256 is in the JSON files and
`gptoss_usage.py` checks it). Every file's SHA-256 is in `MANIFEST.json`.

## License

Code and results: MIT (`LICENSE`). The `gpt-oss-20b` weights belong to OpenAI under Apache-2.0 and are not
redistributed here.
