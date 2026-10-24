# Neural Weight Atlas — gpt-oss-20b num computador comum

*English below.*

Corte guiado por uso dos especialistas (MoE) do `gpt-oss-20b`: de 32 para 20 especialistas por camada, escolhidos pelo
quanto o roteador realmente os usa. O modelo cai de 13,79 GB para 8,29 GB, passa a caber inteiro na GPU de um
MacBook Air M5 com 16 GB e fica 1,6× mais rápido, com o mesmo placar nos testes daqui.

Um comando refaz o modelo a partir do `gpt-oss:20b` oficial e o roda no melhor dispositivo do computador. Os pesos não
estão neste repositório; a receita os refaz byte a byte.

## Usar

```bash
git clone https://github.com/zarustra1997/neural-weight-atlas
cd neural-weight-atlas
python3 nwa.py          # refaz o modelo e o adiciona ao Ollama como gpt-oss-20b-nwa
python3 nwa.py chat     # conversa com ele
python3 nwa.py test     # as 120 perguntas e as 30 novas
```

| Computador | Onde roda | Precisa |
|---|---|---|
| Mac | Ollama direto, na GPU (Metal) | app [Ollama](https://ollama.com/download) e Python 3 |
| PC com placa NVIDIA | Docker, na GPU | Docker com suporte a GPU (NVIDIA Container Toolkit no Linux; Docker Desktop com WSL 2 no Windows) |
| Linux com placa AMD | Docker, na GPU (ROCm) | Docker |
| Outros | Docker, no processador | Docker |

- O `nwa.py` escolhe sozinho. No Mac ele não usa Docker, porque o Docker não alcança a GPU da Apple.
- Memória: o modelo pronto ocupa ~8,4 GB (GPU, RAM ou as duas). Disco: até ~45 GB livres durante a montagem; no fim,
  os dois arquivos em `results/prune/models` (18 GB) podem ser apagados.
- Versões fixas para sair o mesmo arquivo: Ollama 0.40.1 e o llama.cpp que ele traz (commit `631109b3`).

**Conferido:** no Mac, o comando refaz o modelo final byte a byte (SHA-256 `ae3d8644…03df`, 8.289.949.408 bytes). No
Docker (Linux arm64, só processador), o corte e a conversão também saíram idênticos. Rodar o modelo no Docker não
foi testado aqui (a máquina virtual do Docker neste Mac tem 8 GB), nem os perfis NVIDIA e AMD, por falta dessas placas.

## Resultados

| Versão | Arquivo | Memória (`ollama ps`) | GPU | tok/s | 120 perguntas | 30 novas | Situação |
|---|---:|---:|---:|---:|---:|---:|---|
| `gpt-oss:20b` original | 13,79 GB | 13 GB | 82% (swap) | 18,1 | 117 | 30 | referência |
| **K=20 guiado** (`gpt-oss-20b-final`) | **8,29 GB** | **7,8–8,4 GB** | **100%** | **29,6** | **119** | **28** | **resultado** |
| K=16 guiado | 7,02 GB | 7,1 GB | 100% | 30,4 | 111 | — | reprovado |
| K=20 + atenção Q5_0, saída Q5_1, embedding IQ4_NL | 7,58 GB | 7,4 GB | 100% | 22,0 | 112 | 29 | reprovado |

- Nas 150 perguntas: original 147, K=20 guiado 147.
- O original lê ~6,1 MB do SSD por token gerado (não cabe na memória); o K=20 lê ~0,04 MB.
- O ganho de velocidade vem de caber inteiro na GPU, não de fazer menos contas: cada token continua usando 4
  especialistas, e o K=16, menor, não ficou mais rápido que o K=20.
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

## O que já existia e o que é novo aqui

Cortar os especialistas pouco usados de um MoE não é ideia nova:
- Lu et al. (ACL 2024) podam e pulam especialistas depois do treino [1].
- O REAP, da Cerebras (ICLR 2026), pontua cada especialista pelo roteador e pela ativação, e mostra que podar supera
  fundir especialistas [2].
- No próprio gpt-oss-20b, Kapusuzoglu et al. (2026) mostram três coisas [3]:
  - o roteamento é muito espalhado (o mapa daqui confirma: os 20 mais usados cobrem só 87,6% do peso do roteador);
  - a probabilidade do roteador deixa de ser um bom sinal para podar;
  - a perplexidade não prevê o acerto: a poda com menor perplexidade foi a pior em matemática.

O que este repositório acrescenta:
1. **Uma receita conferida para um computador comum.** Ela refaz o arquivo final byte a byte, com versões fixas.
2. **A medida do efeito num Mac de 16 GB.** O corte tira o modelo do swap e o põe inteiro na GPU.
3. **Uma avaliação pré-registrada**, com os resultados negativos e um controle aleatório do mesmo tamanho.
4. **Uma impressão digital linear** para conferir contas com os pesos.

## Limites

- Equivalência formal **não** demonstrada: sem margem de não inferioridade, e os testes estão perto do teto.
- **Matemática com raciocínio longo não foi testada.** É justamente onde [3] achou a maior perda ao podar o gpt-oss-20b
  pelo roteador. Um teste padrão (por exemplo, GSM8K) é o próximo passo.
- A fidelidade (P1) foi medida no corte com atenção e saída em BF16, isolando o efeito do corte; não no arquivo final.
- As medidas foram feitas em uma máquina: MacBook Air M5, 16 GB, macOS, Ollama.
- O mapa está preso ao arquivo do `gpt-oss:20b` com SHA-256 `8ecb5d4a…`. Se a Ollama trocar esse arquivo, o corte
  recusa, em vez de gerar outro modelo.
- O forward NumPy (`gptoss_forward.py`) reproduz o top-1 do llama-server em 6/6 casos e 99,4% das escolhas do roteador,
  mas não passa na regra estrita de ±0,15 em 2 casos em português.
- `gptoss_forward.py` põe no `sys.path` a pasta `../NEURAL_WEIGHT_ATLAS_V1/vendor`, se existir (sobra da máquina
  original; ficou para não mudar o SHA-256). Sem ela, usa o `numpy` instalado.
- A calibração e a P1 usaram prosa local que não está aqui (`EXCLUIDO.md`). Para recalcular o mapa, troque os
  caminhos em `PROSE` por textos seus; o mapa publicado (`results/experts/usage.json`) já basta para refazer o corte.

## Passo a passo (o que o `nwa.py` faz)

```bash
ollama pull gpt-oss:20b                       # o blob deve ter SHA-256 8ecb5d4a…de716
python3 prune_experts.py --label repro --usage results/experts/usage.json --k 20
python3 build_final.py --label repro --ollama-name gpt-oss-20b-nwa
python3 bench_moe.py --target ollama --model gpt-oss-20b-nwa --label nwa
python3 bench_holdout.py --model gpt-oss-20b-nwa=nwa
```

- O corte só lê o original (confere o SHA-256 antes e depois) e deve gerar o intermediário `70df4e5c…317f5`.
- O `build_final.py` usa `MXFP4_MOE`: mantém os especialistas idênticos e põe o resto em Q8_0 (com dimensão 2880,
  quantizações K não servem). O resultado diz se o arquivo saiu idêntico ao publicado.
- Fora do Mac, ou com o Ollama em outro lugar, use `LLAMA_QUANTIZE` (caminho do `llama-quantize`) e `OLLAMA_MODELS`
  (pasta de modelos do Ollama).
- Mapa de uso do zero: `gptoss_calibration.py` → `gptoss_usage.py`, com `numpy` (`requirements.txt`); só CPU, menos
  de 2 GB de RAM.
- Provas: `gptoss_fidelity.py`, `expert_redundancy.py`, `muse_io_limit.py`, `weight_fingerprint.py`.

`gptoss_forward.py` está byte a byte igual ao usado nos resultados (o SHA-256 dele está nos JSON e `gptoss_usage.py`
o confere). Os SHA-256 de cada arquivo desta cópia estão em `MANIFEST.json`.

## Referências

1. Lu et al. *Not All Experts are Equal: Efficient Expert Pruning and Skipping for Mixture-of-Experts Large Language
   Models.* ACL 2024. [arXiv:2402.14800](https://arxiv.org/abs/2402.14800)
2. Lasby et al. *REAP the Experts: Why Pruning Prevails for One-Shot MoE Compression.* ICLR 2026.
   [arXiv:2510.13999](https://arxiv.org/abs/2510.13999)
3. Kapusuzoglu et al. *When Load-Balancing Goes Too Far: Expert Pruning in Over-Dispersed Mixture-of-Experts Models.*
   2026. [arXiv:2609.04453](https://arxiv.org/abs/2609.04453)

## Licença

Código e resultados: MIT (`LICENSE`). Os pesos do `gpt-oss-20b` são da OpenAI, sob Apache-2.0, e não são
redistribuídos aqui.

---

# Neural Weight Atlas — gpt-oss-20b on an ordinary computer

Usage-guided pruning of the `gpt-oss-20b` experts (MoE): from 32 to 20 experts per layer, chosen by how much the router
actually uses them. The model shrinks from 13.79 GB to 8.29 GB, fits entirely on the GPU of a 16 GB MacBook Air M5,
and runs 1.6× faster, with the same score on the tests here.

One command rebuilds the model from the official `gpt-oss:20b` and runs it on the best device the computer has. The
weights are not in this repository; the recipe rebuilds them byte for byte.

## Use

```bash
git clone https://github.com/zarustra1997/neural-weight-atlas
cd neural-weight-atlas
python3 nwa.py          # rebuild the model and add it to Ollama as gpt-oss-20b-nwa
python3 nwa.py chat     # talk to it
python3 nwa.py test     # the 120 questions and the 30 new ones
```

| Computer | Runs on | Needs |
|---|---|---|
| Mac | Ollama directly, on the GPU (Metal) | the [Ollama](https://ollama.com/download) app and Python 3 |
| PC with an NVIDIA card | Docker, on the GPU | Docker with GPU support (NVIDIA Container Toolkit on Linux; Docker Desktop with WSL 2 on Windows) |
| Linux with an AMD card | Docker, on the GPU (ROCm) | Docker |
| Anything else | Docker, on the CPU | Docker |

- `nwa.py` picks on its own. On a Mac it skips Docker, which cannot reach Apple's GPU.
- Memory: the finished model takes ~8.4 GB (GPU, RAM or both). Disk: up to ~45 GB free while building; afterwards the
  two files in `results/prune/models` (18 GB) can be deleted.
- Pinned versions so the same file comes out: Ollama 0.40.1 and the llama.cpp it bundles (commit `631109b3`).

**Checked:** on a Mac, the command rebuilds the final model byte for byte (SHA-256 `ae3d8644…03df`, 8,289,949,408
bytes). In Docker (Linux arm64, CPU only) the cut and the
conversion also came out identical. Running the model in Docker was not tested here (Docker's VM on this Mac has
8 GB), nor were the NVIDIA and AMD profiles, for lack of those cards.

## Results

| Version | File | Memory (`ollama ps`) | GPU | tok/s | 120 questions | 30 new | Status |
|---|---:|---:|---:|---:|---:|---:|---|
| `gpt-oss:20b` original | 13.79 GB | 13 GB | 82% (swap) | 18.1 | 117 | 30 | reference |
| **K=20 guided** (`gpt-oss-20b-final`) | **8.29 GB** | **7.8–8.4 GB** | **100%** | **29.6** | **119** | **28** | **result** |
| K=16 guided | 7.02 GB | 7.1 GB | 100% | 30.4 | 111 | — | rejected |
| K=20 + attention Q5_0, output Q5_1, embedding IQ4_NL | 7.58 GB | 7.4 GB | 100% | 22.0 | 112 | 29 | rejected |

- Over all 150 questions: original 147, K=20 guided 147.
- The original reads ~6.1 MB from the SSD per generated token (it does not fit in memory); K=20 reads ~0.04 MB.
- The speed-up comes from fitting entirely on the GPU, not from doing less math: each token still uses 4 experts, and
  the smaller K=16 was not faster than K=20.
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

## Prior work and what is new here

Pruning rarely used experts from an MoE is not new:
- Lu et al. (ACL 2024) prune and skip experts after training [1].
- Cerebras' REAP (ICLR 2026) scores each expert by router gates and activations, and shows that pruning beats merging
  experts [2].
- On gpt-oss-20b itself, Kapusuzoglu et al. (2026) show three things [3]:
  - routing is highly dispersed (the map here agrees: the top 20 experts cover only 87.6% of the router weight);
  - router probability stops being a good pruning signal;
  - perplexity does not predict accuracy: the lowest-perplexity pruning was the worst at math.

What this repository adds:
1. **A checked recipe for an ordinary computer.** It rebuilds the final file byte for byte, with pinned versions.
2. **The measured effect on a 16 GB Mac.** The cut takes the model out of swap and puts it entirely on the GPU.
3. **A pre-registered evaluation**, with the negative results and a random control of the same size.
4. **A linear weight fingerprint** for checking computations against the weights.

## Limits

- Formal equivalence is **not** shown: no non-inferiority margin, and the tests are near ceiling.
- **Long-reasoning math was not tested.** That is exactly where [3] found the largest loss when pruning gpt-oss-20b by
  the router. A standard benchmark (e.g. GSM8K) is the next step.
- Fidelity (P1) was measured on the cut with BF16 attention and output, isolating the pruning; not on the final file.
- The measurements come from one machine: MacBook Air M5, 16 GB, macOS, Ollama.
- The map is tied to the `gpt-oss:20b` file with SHA-256 `8ecb5d4a…`. If Ollama replaces that file, the cut refuses
  instead of producing a different model.
- The NumPy forward (`gptoss_forward.py`) matches llama-server's top-1 in 6/6 cases and 99.4% of router choices, but
  fails the strict ±0.15 rule in 2 Portuguese cases.
- `gptoss_forward.py` adds `../NEURAL_WEIGHT_ATLAS_V1/vendor` to `sys.path` if it exists (left over from the original
  machine; kept so its SHA-256 does not change). Without it, the installed `numpy` is used.
- Calibration and P1 used local prose that is not included (`EXCLUIDO.md`). To recompute the map, point `PROSE` to
  your own texts; the published map (`results/experts/usage.json`) is enough to redo the cut.

## Step by step (what `nwa.py` does)

The commands are in the Portuguese section above. The cut only reads the original (it checks the SHA-256 before and
after) and should produce the intermediate file `70df4e5c…317f5`. `build_final.py` uses `MXFP4_MOE`, which keeps the
experts identical and puts the rest in Q8_0 (with dimension 2880, K-quants do not apply); its result says whether the
file matches the published one. Outside a Mac, or with Ollama elsewhere, set `LLAMA_QUANTIZE` (path to
`llama-quantize`) and `OLLAMA_MODELS` (Ollama's model folder). The usage map from scratch: `gptoss_calibration.py` →
`gptoss_usage.py`, with `numpy` (`requirements.txt`).

`gptoss_forward.py` is byte-identical to the one used for the results (its SHA-256 is in the JSON files and
`gptoss_usage.py` checks it). Every file's SHA-256 is in `MANIFEST.json`.

## References

See the list in the Portuguese section above: [1] arXiv:2402.14800, [2] arXiv:2510.13999, [3] arXiv:2609.04453.

## License

Code and results: MIT (`LICENSE`). The `gpt-oss-20b` weights belong to OpenAI under Apache-2.0 and are not
redistributed here.
