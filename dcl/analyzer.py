"""Análise de uma avaliação: IA quando há chave, heurística quando não há."""

from __future__ import annotations

import json
from typing import Callable, Iterable

from . import heuristic, llm
from .taxonomy import CATEGORIES, validate_analysis

SYSTEM_PROMPT = f"""Você é analista de reputação de construtoras e incorporadoras brasileiras.
Recebe UMA avaliação de cliente e devolve SOMENTE um objeto JSON, sem texto extra.

Faça análise de sentimento POR ASPECTO: uma avaliação pode elogiar uma coisa e reclamar de outra.

Categorias permitidas (use exatamente estes nomes):
{chr(10).join(f'- "{nome}": {definicao}' for nome, definicao in CATEGORIES.items())}

Formato:
{{
  "sentiment": "Positivo" | "Neutro" | "Negativo",   // tom geral; elogio e crítica equilibrados = "Neutro"
  "aspects": [{{"category": "<categoria>", "sentiment": "Positivo" | "Neutro" | "Negativo"}}],  // 1 a 4 itens, sem repetir categoria
  "urgency": "Baixa" | "Média" | "Alta",
  "legal_risk": true | false,
  "legal_reason": "<por que há risco jurídico, ou vazio>",
  "summary": "<resumo factual em português, no máximo 6 palavras>"
}}

Regras:
- Só inclua aspectos que o texto realmente menciona. Use "Outro" apenas se nada se aplicar.
- urgency "Alta": risco à segurança/saúde (infiltração, vazamento, rachadura, elétrica, elevador, fachada),
  risco jurídico, ou cliente sem resposta há semanas. "Média": reclamação comum. "Baixa": elogio ou neutro.
- legal_risk true quando o cliente cita Procon, advogado, Justiça, processo, notificação, "meus direitos"
  ou disputa de distrato/devolução de valores.
- O resumo descreve o fato ("Atraso de 8 meses na entrega"), não o sentimento ("cliente insatisfeito").

Exemplo
Avaliação: "Excelente atendimento da corretora, mas o reajuste pelo INCC foi bem salgado."
{{"sentiment": "Neutro", "aspects": [{{"category": "Atendimento e vendas", "sentiment": "Positivo"}},
{{"category": "Financeiro e contrato", "sentiment": "Negativo"}}], "urgency": "Baixa", "legal_risk": false,
"legal_reason": "", "summary": "Reajuste INCC alto, bom atendimento"}}
"""


def analyze_review(text: str, use_llm: bool | None = None) -> dict:
    """Devolve a análise normalizada + o motor usado (`engine`)."""
    if use_llm is None:
        use_llm = llm.is_available()
    if use_llm:
        try:
            result = llm.chat_json(SYSTEM_PROMPT, f"Avaliação: {json.dumps(text, ensure_ascii=False)}", validate_analysis)
            return {**result, "engine": f"ia:{llm.get_model()}"}
        except llm.LLMError as exc:
            result = heuristic.analyze(text)
            return {**result, "engine": "heuristica", "warning": f"IA indisponível, usada heurística ({exc})"}
    return {**heuristic.analyze(text), "engine": "heuristica"}


def analyze_many(
    texts: Iterable[str],
    use_llm: bool | None = None,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[dict]:
    texts = list(texts)
    results = []
    for i, text in enumerate(texts, start=1):
        results.append(analyze_review(text, use_llm))
        if on_progress:
            on_progress(i, len(texts))
    return results
