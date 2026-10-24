# O que ficou de fora e por quê

| Fora | Por quê |
|---|---|
| Pesos (GGUF do corte e do modelo final) | São derivados dos pesos da OpenAI e pesam 8–10 GB. A receita e o mapa de uso refazem o arquivo. |
| `results/experts/calibration_tokens_v1.json` e `results/v4/fidelity_tokens.json` | Contêm, em tokens, a prosa usada na calibração e na P1, que inclui documentação de projetos de terceiros. |
| Prosa local da calibração e da P1 (relatórios internos do projeto) | Documentos internos de trabalho. Os scripts indicam os nomes em `PROSE`; troque por textos seus. |
| Integração com um sistema privado (P2 e adaptação) | Depende de código que não é público. |
| Modelfiles, logs do `llama-quantize` e do `llama-server` | Registros da máquina local, com caminhos pessoais; não mudam nenhum número. |
| Atlas visual e painéis HTML | Ferramentas de trabalho locais, não necessárias para reproduzir. |
| Pipeline das versões anteriores (Qwen, V1/V2) | Depende de arquivos dessas versões, que não fazem parte desta publicação. A resposta do Qwen nas 120 perguntas fica como linha de base. |
| Registros internos de coordenação e limpeza | Não fazem parte da pesquisa. |

Nos arquivos copiados, os caminhos pessoais viraram `~` e nomes internos foram neutralizados. Nenhum número foi
alterado. `gptoss_forward.py` está byte a byte igual ao original.
