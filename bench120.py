#!/usr/bin/env python3
"""120 verifiable questions for the original and reduced Qwen3 4B models already in the local Ollama.

Main score = EXACT equality of the normalized first line with an accepted answer (no partial matching;
some items are case-sensitive). The older word-in-first-line rule is kept only for comparison.
Speed is measured separately with long answers (128 tokens), alternating models. Writes only results/bench120/.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'results' / 'bench120'
OLLAMA = 'http://127.0.0.1:11434'
MODELS = ('qwen3:4b-instruct-2507-q4_K_M', 'qwen3-4b-reduzido-teste', 'qwen3-4b-iq2m-teste')
BASELINE = MODELS[0]
SEED = 20261017
ANSWER_TOKENS = 64
CONTEXT = 4096
SPEED_TOKENS = 128
SPEED_REPEATS = 3
TIMEOUT_SECONDS = 600
WILSON_Z = 1.96
N, W, NAME = ' Responda só com o número.', ' Responda com uma palavra.', ' Responda só com o nome.'
YES_NO = ' Responda só SIM ou NÃO.'
SPEED_PROMPTS = ('Escreva um parágrafo de cerca de 150 palavras explicando como funciona um avião.',
                 'Explique em detalhes, em português, o que é a memória RAM de um computador.',
                 'Write about 150 words describing the water cycle.')

# (category, question, accepted answers, case_sensitive)
QUESTIONS = [
    ('matemática', 'Quanto é 48 + 67?' + N, ['115'], False),
    ('matemática', 'Quanto é 1000 − 357?' + N, ['643'], False),
    ('matemática', 'Quanto é 25 × 16?' + N, ['400'], False),
    ('matemática', 'Quanto é 144 ÷ 9?' + N, ['16'], False),
    ('matemática', 'Quanto é 13 × 13?' + N, ['169'], False),
    ('matemática', 'Quanto é 2 elevado a 8?' + N, ['256'], False),
    ('matemática', 'Quanto é 3 elevado a 4?' + N, ['81'], False),
    ('matemática', 'Qual é a raiz quadrada de 225?' + N, ['15'], False),
    ('matemática', 'Quanto é 20% de 350?' + N, ['70'], False),
    ('matemática', 'Quanto é 7 × 6 + 5?' + N, ['47'], False),
    ('matemática', 'Quanto é (12 + 8) × 3?' + N, ['60'], False),
    ('matemática', 'Um produto custa 80 reais e tem desconto de 25%. Qual é o preço final em reais?' + N, ['60'], False),
    ('matemática', 'Quantos segundos há em 3 minutos?' + N, ['180'], False),
    ('matemática', 'Quantos minutos há em 4 horas?' + N, ['240'], False),
    ('matemática', 'João tem 6 anos. Ana tem 5 vezes a idade de João. Quantos anos Ana tem?' + N, ['30'], False),
    ('matemática', 'Qual é o próximo número da sequência 3, 6, 9, 12?' + N, ['15'], False),
    ('matemática', 'Qual é o próximo número da sequência 1, 4, 9, 16, 25?' + N, ['36'], False),
    ('matemática', 'Um retângulo tem lados 7 e 9. Qual é a sua área?' + N, ['63'], False),
    ('matemática', 'Um quadrado tem perímetro 36. Quanto mede cada lado?' + N, ['9'], False),
    ('matemática', 'Quanto é 0,5 + 0,25?' + N, ['0,75', '0.75'], False),
    ('matemática', 'Quanto é um quarto de 120?' + N, ['30'], False),
    ('matemática', 'Se x + 7 = 19, quanto vale x?' + N, ['12', 'x=12', 'x = 12'], False),
    ('matemática', 'Se 3x = 27, quanto vale x?' + N, ['9', 'x=9', 'x = 9'], False),
    ('matemática', 'Quantos dias há em 3 semanas?' + N, ['21'], False),
    ('matemática', 'Um carro faz 12 km por litro. Quantos km ele faz com 15 litros?' + N, ['180'], False),
    ('lógica', 'Todos os gatos são mamíferos. Mimi é um gato. Mimi é um mamífero?' + YES_NO, ['sim'], False),
    ('lógica', 'Se chove, a rua fica molhada. A rua não está molhada. Choveu?' + YES_NO, ['não', 'nao'], False),
    ('lógica', 'Pedro é mais velho que Ana. Ana é mais velha que Luís. Quem é o mais velho?' + NAME, ['pedro'], False),
    ('lógica', 'Pedro é mais velho que Ana. Ana é mais velha que Luís. Quem é o mais novo?' + NAME, ['luís', 'luis'], False),
    ('lógica', 'Hoje é quarta-feira. Que dia da semana foi ontem?' + W, ['terça', 'terça-feira'], False),
    ('lógica', 'Hoje é sexta-feira. Que dia da semana será daqui a 2 dias?' + W, ['domingo'], False),
    ('lógica', 'O dia 1 de um mês caiu numa segunda-feira. Em que dia da semana cai o dia 8 desse mês?' + W, ['segunda', 'segunda-feira'], False),
    ('lógica', 'Numa fila, Bia está na frente de Caio e atrás de Dora. Quem está no meio?' + NAME, ['bia'], False),
    ('lógica', 'Se A é maior que B e B é maior que C, então A é maior que C?' + YES_NO, ['sim'], False),
    ('lógica', 'Qual palavra não pertence ao grupo: maçã, banana, cenoura, uva?' + W, ['cenoura'], False),
    ('lógica', 'Qual palavra não pertence ao grupo: carro, ônibus, bicicleta, mesa?' + W, ['mesa'], False),
    ('lógica', 'Nenhum peixe voa. O salmão é um peixe. O salmão voa?' + YES_NO, ['não', 'nao'], False),
    ('lógica', 'Uma caixa só tem bolas vermelhas. Se eu tirar uma bola, de que cor ela será?' + W, ['vermelha', 'vermelho'], False),
    ('lógica', 'Se 5 máquinas fazem 5 peças em 5 minutos, quantos minutos 1 máquina leva para fazer 1 peça?' + N, ['5'], False),
    ('lógica', 'Maria tem 4 canetas. João tem o dobro de canetas de Maria. Quantas canetas os dois têm juntos?' + N, ['12'], False),
    ('lógica', 'Um relógio marca 10 horas. Que hora marcará 5 horas depois, no formato de 0 a 23?' + N, ['15'], False),
    ('lógica', "Qual é o oposto de 'norte'?" + W, ['sul'], False),
    ('lógica', 'O dia 1 de um mês é um domingo. Em que dia da semana cai o dia 15?' + W, ['domingo'], False),
    ('lógica', 'Lucas é mais alto que Rafa. Rafa é mais alto que Téo. Téo é mais alto que Lucas?' + YES_NO, ['não', 'nao'], False),
    ('lógica', 'Coloque em ordem crescente e responda só com os números separados por vírgula: 9, 2, 7', ['2, 7, 9'], False),
    ('fatos', 'Qual é a capital da Itália?' + W, ['roma'], False),
    ('fatos', 'Qual é a capital da Argentina?' + NAME, ['buenos aires'], False),
    ('fatos', 'Qual é a capital do Canadá?' + W, ['ottawa', 'otava'], False),
    ('fatos', 'Qual é o maior oceano da Terra?' + W, ['pacífico', 'pacifico'], False),
    ('fatos', 'Qual planeta é conhecido como planeta vermelho?' + W, ['marte'], False),
    ('fatos', 'Qual é o símbolo químico do ferro? Responda só com o símbolo.', ['fe'], False),
    ('fatos', 'Qual é o símbolo químico do sódio? Responda só com o símbolo.', ['na'], False),
    ('fatos', 'Quantos lados tem um triângulo?' + N, ['3'], False),
    ('fatos', 'Em que ano o ser humano pisou na Lua pela primeira vez?' + N, ['1969'], False),
    ('fatos', 'Quem pintou a Mona Lisa?' + NAME, ['leonardo da vinci', 'da vinci'], False),
    ('fatos', 'Qual gás as plantas absorvem do ar na fotossíntese? Responda só com o nome do gás.',
     ['dióxido de carbono', 'dioxido de carbono', 'gás carbônico', 'gas carbonico', 'co2', 'co₂'], False),
    ('fatos', 'Qual é o maior país do mundo em área?' + W, ['rússia', 'russia'], False),
    ('fatos', 'Qual é o idioma oficial do Brasil?' + W, ['português', 'portugues'], False),
    ('fatos', 'Quantas horas tem um dia?' + N, ['24'], False),
    ('fatos', 'Qual é o ponto de ebulição da água ao nível do mar, em graus Celsius?' + N, ['100'], False),
    ('fatos', 'Qual é o menor número primo?' + N, ['2'], False),
    ('fatos', "Quem escreveu 'Os Lusíadas'?" + NAME, ['luís de camões', 'luis de camoes', 'camões', 'camoes', 'luís vaz de camões', 'luis vaz de camoes'], False),
    ('fatos', 'Qual metal é líquido à temperatura ambiente?' + W, ['mercúrio', 'mercurio'], False),
    ('fatos', 'Qual é a moeda do Japão?' + W, ['iene', 'yen'], False),
    ('fatos', 'Qual é o satélite natural da Terra?' + W, ['lua'], False),
    ('interpretação', "Texto: 'Marta comprou 3 maçãs e 2 peras. Depois comeu uma maçã.' Quantas maçãs Marta tem agora?" + N, ['2'], False),
    ('interpretação', "Texto: 'O trem saiu às 8h e chegou às 11h.' Quantas horas durou a viagem?" + N, ['3'], False),
    ('interpretação', "Texto: 'Carlos mora em Recife e trabalha em Olinda.' Em que cidade Carlos mora?" + W, ['recife'], False),
    ('interpretação', "Texto: 'A reunião foi adiada de terça para quinta.' Em que dia será a reunião?" + W, ['quinta', 'quinta-feira'], False),
    ('interpretação', "Texto: 'Lia é mais rápida que Bruno, mas mais lenta que Eva.' Quem é a mais rápida?" + NAME, ['eva'], False),
    ('interpretação', "Texto: 'O livro tem 200 páginas. Ana leu metade.' Quantas páginas Ana leu?" + N, ['100'], False),
    ('interpretação', "Texto: 'Choveu de manhã, mas à tarde fez sol.' Como estava o tempo à tarde?" + W, ['sol', 'ensolarado', 'ensolarada'], False),
    ('interpretação', "Texto: 'Paulo tinha 50 reais e gastou 18.' Quantos reais sobraram?" + N, ['32'], False),
    ('interpretação', "Texto: 'O cachorro de Júlia se chama Rex e o gato se chama Tom.' Qual é o nome do gato?" + NAME, ['tom'], False),
    ('interpretação', "Texto: 'A loja abre às 9h e fecha às 18h.' A que horas a loja fecha?" + N, ['18'], False),
    ('interpretação', "Texto: 'Três amigos dividiram igualmente uma conta de 90 reais.' Quanto cada um pagou?" + N, ['30'], False),
    ('interpretação', "Texto: 'Renata nasceu em 1990.' Quantos anos ela completou em 2020?" + N, ['30'], False),
    ('interpretação', "Texto: 'O carro azul é de Pedro; o vermelho é de Sara.' De quem é o carro vermelho?" + NAME, ['sara'], False),
    ('interpretação', "Texto: 'Eu tinha 4 lápis, perdi 1 e ganhei 3.' Quantos lápis eu tenho?" + N, ['6'], False),
    ('interpretação', "Texto: 'A festa começa às 20h e dura 3 horas.' A que horas a festa termina?" + N, ['23'], False),
    ('instruções', 'Escreva apenas a palavra AZUL em letras maiúsculas.', ['AZUL'], True),
    ('instruções', 'Responda apenas com o número 7, sem mais nada.', ['7'], False),
    ('instruções', "Escreva a palavra 'casa' de trás para frente. Responda só com o resultado.", ['asac'], False),
    ('instruções', 'Escreva os números de 1 a 5 separados por vírgula, sem mais nada.', ['1, 2, 3, 4, 5'], False),
    ('instruções', "Escreva a palavra 'gato' três vezes, separadas por um espaço, sem mais nada.", ['gato gato gato'], False),
    ('instruções', 'Responda só com a letra A maiúscula.', ['A'], True),
    ('instruções', 'Qual é a primeira letra do alfabeto? Responda só com a letra, em minúscula.', ['a'], True),
    ('instruções', "Escreva a sigla de 'Organização das Nações Unidas'. Responda só com a sigla.", ['ONU'], True),
    ('instruções', "Qual é a tradução de 'cachorro' para o inglês? Responda com uma única palavra.", ['dog'], False),
    ('instruções', 'Escreva o número dez em algarismos. Responda só com o número.', ['10'], False),
    ('instruções', 'Escreva o número 15 por extenso. Responda só com a palavra.', ['quinze'], False),
    ('instruções', "Escreva a palavra 'Brasil' toda em letras minúsculas. Responda só com a palavra.", ['brasil'], True),
    ('instruções', "Responda apenas 'ok', sem mais nada.", ['ok'], False),
    ('instruções', 'Some 2 e 3 e responda exatamente no formato resultado=N, sem espaços.', ['resultado=5'], False),
    ('instruções', 'Escreva as cinco vogais em ordem alfabética, sem espaços nem vírgulas.', ['aeiou'], False),
    ('instruções', "Escreva a palavra 'sol' entre colchetes, sem mais nada.", ['[sol]'], False),
    ('instruções', "Converta 'BOM DIA' para letras minúsculas. Responda só com o resultado.", ['bom dia'], True),
    ('instruções', 'Qual dia da semana vem depois de segunda-feira?' + W, ['terça', 'terça-feira'], False),
    ('instruções', "Escreva 'sim' se 10 for maior que 3; caso contrário, escreva 'não'. Responda só com a palavra.", ['sim'], False),
    ('instruções', "Escreva apenas o último caractere da palavra 'mesa'.", ['a'], False),
    ('português', "Qual é o plural de 'cidadão'?" + W, ['cidadãos'], False),
    ('português', "Qual é o plural de 'papel'?" + W, ['papéis'], False),
    ('português', "Qual é o feminino de 'ator'?" + W, ['atriz'], False),
    ('português', "Qual é o antônimo de 'alto'?" + W, ['baixo'], False),
    ('português', "Qual é o diminutivo de 'livro'?" + W, ['livrinho'], False),
    ('português', "Complete com o verbo 'ir' no pretérito perfeito: 'Eu ___ ao mercado ontem.' Responda só com o verbo.", ['fui'], False),
    ('português', "Qual é o pretérito perfeito de 'eu faço'? Responda só com a forma verbal.", ['fiz', 'eu fiz'], False),
    ('português', "Quantas sílabas tem a palavra 'janela'?" + N, ['3'], False),
    ('português', "Qual é o substantivo coletivo de 'abelhas'?" + W, ['enxame'], False),
    ('português', "Qual é o antônimo de 'feliz'?" + W, ['triste', 'infeliz'], False),
    ('inglês', 'What is 9 times 7? Answer with only the number.', ['63'], False),
    ('inglês', 'What is the capital of Germany? Answer with one word.', ['berlin'], False),
    ('inglês', 'What color do you get by mixing blue and yellow? Answer with one word.', ['green'], False),
    ('inglês', "Translate 'livro' into English. Answer with one word.", ['book'], False),
    ('inglês', "What is the plural of 'child'? Answer with one word.", ['children'], False),
    ('inglês', "What is the past tense of 'go'? Answer with one word.", ['went'], False),
    ('inglês', 'How many days are in a week? Answer with only the number.', ['7'], False),
    ('inglês', "What is the opposite of 'hot'? Answer with one word.", ['cold'], False),
    ('inglês', "Which animal says 'meow'? Answer with one word.", ['cat'], False),
    ('inglês', "Spell the word 'dog' backwards. Answer with only the result.", ['god'], False),
]


def first_line(text):
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines[0] if lines else ''


def normalize(text, case_sensitive):
    """Formatting only: markdown bold/code, surrounding spaces, final period or exclamation, spaces around commas."""
    t = text.replace('**', '').replace('`', '').strip()
    t = re.sub(r'[.!]+$', '', t).strip()
    t = re.sub(r'\s*,\s*', ', ', t)
    t = re.sub(r'\s+', ' ', t)
    return t if case_sensitive else t.lower()


def exact(answer, accepted, case_sensitive):
    got = normalize(first_line(answer), case_sensitive)
    return any(got == normalize(a, case_sensitive) for a in accepted)


def word_in_first_line(answer, accepted):
    """Older, looser rule (kept for comparison only)."""
    line = normalize(first_line(answer), False)
    return any(re.search(rf'(?<![\w.]){re.escape(normalize(a, False))}(?![\w])', line) for a in accepted)


def chat(model, prompt, tokens):
    body = json.dumps({'model': model, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False,
                       'options': {'temperature': 0, 'seed': SEED, 'num_predict': tokens, 'num_ctx': CONTEXT}}).encode()
    req = urllib.request.Request(OLLAMA + '/api/chat', body, {'Content-Type': 'application/json'})
    start = time.perf_counter()
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as r:
        data = json.loads(r.read())
    data['wall_seconds'] = time.perf_counter() - start
    return data


def wilson(k, n):
    p = k / n
    centre = (p + WILSON_Z ** 2 / (2 * n)) / (1 + WILSON_Z ** 2 / n)
    half = WILSON_Z * math.sqrt(p * (1 - p) / n + WILSON_Z ** 2 / (4 * n * n)) / (1 + WILSON_Z ** 2 / n)
    return [round(centre - half, 3), round(centre + half, 3)]


def mcnemar(base, other):
    lost = sum(b and not o for b, o in zip(base, other))
    gained = sum(o and not b for b, o in zip(base, other))
    n = lost + gained
    p = min(1.0, 2 * sum(math.comb(n, j) for j in range(min(lost, gained) + 1)) / 2 ** n) if n else 1.0
    return {'lost_vs_original': lost, 'gained_vs_original': gained, 'exact_two_sided_p': p}


def run_questions(model, questions):
    chat(model, 'Responda só: ok', 4)  # warm-up / load, not scored
    rows = []
    for q in questions:
        d = chat(model, q['question'], ANSWER_TOKENS)
        answer = d['message']['content']
        rows.append({'id': q['id'], 'category': q['category'], 'answer': answer,
                     'exact': exact(answer, q['accept'], q['case_sensitive']), 'word_rule': word_in_first_line(answer, q['accept']),
                     'eval_count': d.get('eval_count'), 'wall_seconds': round(d['wall_seconds'], 3)})
    return rows


def run_speed():
    """Long answers, models alternated round by round; eval rate from Ollama's own counters."""
    samples = {m: [] for m in MODELS}
    for m in MODELS:
        chat(m, 'Responda só: ok', 4)
    for _ in range(SPEED_REPEATS):
        for m in MODELS:
            for prompt in SPEED_PROMPTS:
                d = chat(m, prompt, SPEED_TOKENS)
                samples[m].append({'eval_count': d['eval_count'], 'tokens_per_second': d['eval_count'] / d['eval_duration'] * 1e9,
                                   'prompt_tokens_per_second': d['prompt_eval_count'] / d['prompt_eval_duration'] * 1e9
                                   if d.get('prompt_eval_duration') else None})
    return {m: {'median_tokens_per_second': statistics.median(s['tokens_per_second'] for s in v),
                'min': min(s['tokens_per_second'] for s in v), 'max': max(s['tokens_per_second'] for s in v),
                'median_tokens_generated': statistics.median(s['eval_count'] for s in v), 'samples': v}
            for m, v in samples.items()}


def run():
    started = time.perf_counter()
    OUT.mkdir(parents=True, exist_ok=True)
    questions = [{'id': i + 1, 'category': c, 'question': q, 'accept': a, 'case_sensitive': cs}
                 for i, (c, q, a, cs) in enumerate(QUESTIONS)]
    (OUT / 'questions.json').write_text(json.dumps(questions, ensure_ascii=False, indent=1) + '\n')
    results = {}
    for model in MODELS:
        rows = run_questions(model, questions)
        (OUT / f"answers_{model.replace(':', '_')}.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1) + '\n')
        results[model] = rows
        print(model, sum(r['exact'] for r in rows), '/', len(rows), flush=True)
    speed = run_speed()
    base = [r['exact'] for r in results[BASELINE]]
    summary = {'questions': len(questions), 'models': {}, 'speed': speed,
               'settings': {'temperature': 0, 'seed': SEED, 'num_predict': ANSWER_TOKENS, 'num_ctx': CONTEXT,
                            'speed_tokens': SPEED_TOKENS, 'speed_repeats': SPEED_REPEATS},
               'scoring': 'exact = equality of the normalized first line; word_rule = older looser rule, comparison only'}
    for model, rows in results.items():
        k = sum(r['exact'] for r in rows)
        cats = {}
        for r in rows:
            c = cats.setdefault(r['category'], [0, 0])
            c[0] += r['exact']
            c[1] += 1
        summary['models'][model] = {'exact': k, 'exact_wilson_95': wilson(k, len(rows)),
                                    'word_rule': sum(r['word_rule'] for r in rows),
                                    'by_category': {c: f'{a}/{b}' for c, (a, b) in cats.items()},
                                    'vs_original': mcnemar(base, [r['exact'] for r in rows]) if model != BASELINE else None,
                                    'answers_differ_from_original': sum(r['answer'].strip() != b['answer'].strip()
                                                                        for r, b in zip(rows, results[BASELINE]))}
    summary['seconds'] = round(time.perf_counter() - started, 1)
    (OUT / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1) + '\n')
    print(json.dumps({m: {k: v for k, v in s.items() if k != 'by_category'} for m, s in summary['models'].items()}, indent=1))
    print(json.dumps({m: {k: round(v, 1) if isinstance(v, float) else v for k, v in s.items() if k != 'samples'} for m, s in speed.items()}, indent=1))


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    run()
