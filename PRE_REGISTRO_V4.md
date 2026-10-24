> Versão pública do pré-registro de 20/10/2026: referências internas e nomes de pastas privadas foram removidos; critérios, números e resultados estão inalterados. Os hashes curtos citados são dos arquivos originais; os desta cópia estão em MANIFEST.json.

# NWA V4 — provas pré-registradas (20/10/2026)

Objetivo: provar ou refutar os 4 pontos que ainda não estavam provados. Os critérios abaixo foram escritos ANTES de rodar e não mudam depois. Os resultados entram na seção do fim, mesmo que sejam negativos.

Comparação: `gpt-oss:20b` (original) × `gpt-oss-20b-final` (K=20 guiado, blob `ae3d8644…`). Nada é instalado ou baixado e só um processo pesado roda por vez.

## P1 — Fidelidade palavra a palavra (o que dá para provar no lugar de "equivalente em tudo")

"Equivalente em tudo" não tem prova finita. O que se mede é quanto o corte muda a distribuição da próxima palavra em texto que o corte NÃO viu.

- **Texto:**
  - as 150 conversas que o PRÓPRIO original gerou nos testes (120 + 30), com pergunta, raciocínio e resposta, no formato harmony;
  - prosa pt/en não usada na calibração (3 relatórios internos do projeto e 2 documentos de projetos de terceiros; não incluídos, ver EXCLUIDO.md).
- **Método:**
  - o forward NumPy validado (`gptoss_forward.py`, sem edição), mais uma cópia do laço que devolve todas as posições;
  - o corte é aplicado bloqueando no roteador os especialistas removidos, o que equivale ao arquivo podado;
  - fica isolado o efeito do corte (atenção e saída em BF16 nos dois).
- **Variantes:**
  - original (32);
  - K=20 guiado (o final);
  - K=16 guiado (controle negativo, já reprovado em 111/120);
  - K=20 aleatório, com semente 20261020 (controle: mostra o valor do mapa).
- **Métricas:** em até 3.000 posições sorteadas (semente fixa), a KL(original ‖ variante) média em nats, a concordância da palavra mais provável (top-1), e o mesmo só nas posições geradas pelo modelo.
- **Critério:** "fidelidade alta comprovada" se, no K=20 guiado:
  - KL média ≤ 0,05 nat;
  - top-1 ≥ 95%;
  - KL ao menos 2× menor que a do K=16 e que a do K=20 aleatório.

  Referência (de memória, a verificar): no llama.cpp, quantizações de 4–5 bits consideradas quase sem perda ficam na faixa de 0,01–0,05 nat. Fora disso, o resultado é relatado como está.

## P2 — Avaliação como professor de um sistema privado

Depende de código privado que não está neste repositório.

## P3 — Muse 816B neste Mac (o que dá para provar: o limite)

Rodar o Muse exige baixar 175,6 GB e instalar um llama.cpp novo (fora deste escopo). O que se prova aqui é o teto físico, medido.

- **Simulação:** leitura do SSD com o cache desligado (F_NOCACHE), em pedaços do tamanho de um especialista do Muse a ~1,56 bit (≈ 9,8 MB ESTIMADO), em posições aleatórias dos arquivos grandes que já existem, com 1, 4 e 8 leituras em paralelo.
- **Tokens/s máximos:** GB/s medido ÷ 4,9 GB por token (8 especialistas × 62 camadas, sem acerto de cache).
- **Critério:** a conta de "~1–2 tok/s" fica confirmada se o teto medido ficar entre 0,5 e 2,5 tok/s. Se der mais de 5 tok/s, a conclusão anterior está errada e é revista.

## P4 — Redundância entre especialistas (hipótese A da V4)

- **Dados:** 8 camadas (0, 3, 6, 9, 12, 15, 18, 23) × 3 matrizes (gate, up, down), cada uma com os 32 especialistas desquantizados do blob original.
- **Medidas:**
  1. energia explicada pela média dos 32;
  2. energia que um subespaço COMPARTILHADO de posto r captura, contra a energia que cada especialista captura no SEU próprio subespaço de posto r e contra a linha isotrópica r/2880;
  3. erro RMS relativo da forma W_i ≈ A_i·B no orçamento de memória do K=20 (r = 1745).
- **Critério de sucesso:** subespaço compartilhado ≥ linha isotrópica + 15 pontos, e erro RMS relativo ≤ 0,10 em r = 1745.
- **Critério de abandono:** média < 5% da energia, e subespaço compartilhado ≤ linha isotrópica + 5 pontos.

## Resultados

(preenchido depois de cada prova, com os hashes)

### P4 — Redundância entre especialistas: o critério de sucesso FALHOU por larga margem (veredito formal: INCONCLUSIVO)

Arquivos: `expert_redundancy.py` (00a248e5462a) e `results/v4/expert_redundancy.json` (a5480a297e7d). 24 matrizes, 242 s, pico de 1,6 GB.

| Medida | Resultado | Critério |
|---|---|---|
| Energia explicada pela média dos 32 | 3,1–4,8% | abandono se < 5% → atingido |
| Subespaço compartilhado em r = 1745 | 71–90% contra 60,6% isotrópico (+10 a +29 pontos) | sucesso ≥ +15 em todas; abandono ≤ +5 → nenhum dos dois |
| Erro RMS relativo de A_i·B no orçamento do K=20 | 0,32–0,54 | sucesso ≤ 0,10 → falhou (3 a 5 vezes acima) |
| Cada especialista no seu próprio posto | erro 0,22 em r = 1440 (o ponto em que posto baixo começa a economizar) | sem ganho |

**Leitura:**
- Os especialistas do gpt-oss são quase independentes. Há alguma estrutura compartilhada na entrada, mas longe de permitir guardar 32 na memória de 20 sem erro grande.
- A hipótese A da V4, na forma exata (reconstruir os pesos), está refutada para o gpt-oss.
- Ficam fora deste teste os métodos que ajustam o comportamento em vez de reconstruir os pesos (MoBE/D²-MoE com otimização). Eles exigiriam calibração ou treino.

### P2 — NÃO EXECUTADA

Nenhum resultado foi gravado.

### P1 — Fidelidade palavra a palavra: NÃO ATINGIU O CRITÉRIO (o corte muda a distribuição de forma mensurável)

Arquivos: `gptoss_fidelity.py` (19354d0ba952), `results/v4/fidelity_tokens.json` (2159eee31d43) e `results/v4/fidelity.json` (42ffe5e9e949).

**Execução:** 22.861 tokens (150 conversas do original + 10 trechos de prosa), 3.000 posições sorteadas, das quais 519 foram geradas pelo modelo. 1.507 s, pico de 1,25 GB.

**Auto-conferência:** o laço da prova deu diferença máxima 0,0 contra o `gptoss_forward.forward`.

| Variante | KL média (nat) | Mediana | p95 | Top-1 igual | Só posições geradas: KL / top-1 |
|---|---:|---:|---:|---:|---:|
| **K=20 guiado (final)** | **0,284** | 0,128 | 0,95 | **77,4%** | **0,126 / 91,3%** |
| K=16 guiado | 0,598 | 0,285 | 2,12 | 65,3% | 0,227 / 85,9% |
| K=20 aleatório | 0,915 | 0,668 | 2,92 | 57,0% | 0,313 / 85,0% |

**Critério:** KL ≤ 0,05, top-1 ≥ 95% e KL ≥ 2× menor que os controles.
- A parte relativa foi cumprida: o guiado é 2,1× melhor que o K=16 e 3,2× melhor que o aleatório.
- A parte absoluta não: a KL é 5,7× acima do limite e o top-1 fica 18 pontos abaixo.

**Leitura:**
- O K=20 empata no placar (147/150), mas NÃO é o mesmo modelo palavra a palavra. Uma KL de 0,28 nat fica na faixa de uma quantização agressiva, não de 4–5 bits.
- No texto que ele mesmo gera (raciocínio e resposta) a diferença é menor: 91% de top-1 e KL 0,13.
- O placar de perguntas curtas esconde essa mudança. Isso corrige a leitura "equivalente": o certo é "acerta o mesmo nestes testes, mas é um modelo diferente".
- O que fica PROVADO: o mapa de uso escolhe bem. Com o mesmo tamanho, o corte guiado muda 3× menos o modelo que um corte aleatório.

### P3 — Limite do Muse neste Mac: CONFIRMADO (1,1–1,6 tok/s de teto, sem cache)

Arquivos: `muse_io_limit.py` (1f7012559c93), `results/v4/muse_io_limit.json` (cce5682ac2a9) e `results/v4/muse_io_limit_robustez.json` (fcb67052993a).

**Medido** com SSD frio (F_NOCACHE), em leituras aleatórias de 9,86 MB (um especialista do Muse a ~1,56 bit, tamanho ESTIMADO), 496 por token:

| Leituras em paralelo | GB/s | Teto de tok/s |
|---:|---:|---:|
| 1 | 5,2–5,35 | 1,06–1,09 |
| 4 | 7,3 | 1,49 |
| 8 | 7,5–8,0 | 1,54–1,64 |

A repetição sem o blob possivelmente em cache deu o mesmo.

**Com cache** (ESTIMADO): 30% de acertos daria 2,2 tok/s; 50%, 3,1 tok/s; 80%, 7,7 tok/s. O uso medido no gpt-oss é espalhado, então acerto alto é improvável.

**Conclusão:** o Muse comprimido neste Mac ficaria em ~1–1,6 tok/s no melhor caso, contando só a leitura (sem cálculo nem desquantização). O Muse exato não cabe no disco (1,64 TB contra 494 GB). Isso está provado pela medição.
