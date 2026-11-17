"""Taxonomia única usada pela IA, pela heurística, pelo painel e pela avaliação.

Manter tudo aqui garante que o modelo, os gráficos e as métricas falem a mesma
língua: se uma categoria muda, muda em um lugar só.
"""

from __future__ import annotations

import unicodedata

SENTIMENTS = ["Positivo", "Neutro", "Negativo"]
URGENCY_LEVELS = ["Baixa", "Média", "Alta"]

# Categoria -> definição (a definição vai no prompt do modelo e na aba "Como funciona").
CATEGORIES: dict[str, str] = {
    "Prazo de entrega": "Cumprimento do cronograma, atrasos, adiamentos, entrega no prazo ou antes.",
    "Acabamento e qualidade": "Qualidade construtiva e de materiais: piso, esquadrias, rachaduras, infiltrações, projeto e planta.",
    "Assistência técnica": "Pós-obra e garantia: vistoria, chamados, reparos, tempo de resposta da assistência.",
    "Atendimento e vendas": "Relacionamento: corretores, stand, SAC, comunicação, retorno de contato.",
    "Financeiro e contrato": "Valores, taxas (evolução de obra, transferência), reajustes (INCC), boletos, juros, distrato, cláusulas.",
    "Área comum e infraestrutura": "Lazer, piscina, academia, elevadores, garagem, portaria, paisagismo, áreas do condomínio.",
    "Localização": "Bairro, acesso, vizinhança, proximidade de serviços.",
    "Outro": "Nada acima se aplica ou a avaliação é genérica.",
}
CATEGORY_NAMES = list(CATEGORIES)

# Variações que modelos de linguagem costumam devolver -> forma canônica.
_SENTIMENT_ALIASES = {
    "positivo": "Positivo", "positive": "Positivo", "pos": "Positivo",
    "neutro": "Neutro", "neutral": "Neutro", "misto": "Neutro", "mixed": "Neutro",
    "negativo": "Negativo", "negative": "Negativo", "neg": "Negativo",
}
_URGENCY_ALIASES = {
    "baixa": "Baixa", "low": "Baixa",
    "media": "Média", "medium": "Média", "moderada": "Média",
    "alta": "Alta", "high": "Alta", "critica": "Alta", "urgente": "Alta",
}
_CATEGORY_ALIASES = {
    "prazo": "Prazo de entrega", "atraso de obra": "Prazo de entrega", "atraso": "Prazo de entrega",
    "acabamento": "Acabamento e qualidade", "qualidade": "Acabamento e qualidade",
    "assistencia": "Assistência técnica", "pos-obra": "Assistência técnica", "garantia": "Assistência técnica",
    "atendimento": "Atendimento e vendas", "vendas": "Atendimento e vendas",
    "financeiro/contrato": "Financeiro e contrato", "financeiro": "Financeiro e contrato", "contrato": "Financeiro e contrato",
    "area comum": "Área comum e infraestrutura", "infraestrutura": "Área comum e infraestrutura", "lazer": "Área comum e infraestrutura",
    "localizacao": "Localização",
    "outro": "Outro", "outros": "Outro",
}


def fold(text: str) -> str:
    """Minúsculas e sem acentos, para comparações tolerantes."""
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c)).casefold().strip()


_CANON_BY_FOLD = {fold(c): c for c in CATEGORY_NAMES}


def normalize_sentiment(value) -> str:
    key = fold(value)
    if key in _SENTIMENT_ALIASES:
        return _SENTIMENT_ALIASES[key]
    raise ValueError(f"sentimento inválido: {value!r} (use {', '.join(SENTIMENTS)})")


def normalize_urgency(value) -> str:
    key = fold(value)
    if key in _URGENCY_ALIASES:
        return _URGENCY_ALIASES[key]
    raise ValueError(f"urgência inválida: {value!r} (use {', '.join(URGENCY_LEVELS)})")


def normalize_category(value) -> str:
    key = fold(value)
    if key in _CANON_BY_FOLD:
        return _CANON_BY_FOLD[key]
    if key in _CATEGORY_ALIASES:
        return _CATEGORY_ALIASES[key]
    for alias, canon in _CATEGORY_ALIASES.items():
        if alias in key:
            return canon
    return "Outro"


def to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return fold(value) in {"true", "sim", "yes", "1", "verdadeiro"}


def validate_analysis(raw: dict) -> dict:
    """Valida e normaliza a saída da IA (ou da heurística).

    Levanta ValueError com uma mensagem clara quando algo não dá para salvar;
    essa mensagem é devolvida ao modelo na tentativa seguinte.
    """
    if not isinstance(raw, dict):
        raise ValueError("a resposta precisa ser um objeto JSON")

    sentiment = normalize_sentiment(raw.get("sentiment", ""))

    aspects_in = raw.get("aspects") or []
    if not isinstance(aspects_in, list):
        raise ValueError("'aspects' precisa ser uma lista")
    merged: dict[str, str] = {}
    for item in aspects_in:
        if not isinstance(item, dict):
            continue
        cat = normalize_category(item.get("category", "Outro"))
        try:
            sent = normalize_sentiment(item.get("sentiment", sentiment))
        except ValueError:
            sent = sentiment
        # Se a mesma categoria aparece duas vezes com sinais opostos, vira Neutro.
        merged[cat] = sent if cat not in merged or merged[cat] == sent else "Neutro"
    if not merged:
        merged = {"Outro": sentiment}
    if len(merged) > 1:
        merged.pop("Outro", None)
    aspects = [{"category": c, "sentiment": s} for c, s in merged.items()]

    legal_risk = to_bool(raw.get("legal_risk", False))
    legal_reason = str(raw.get("legal_reason") or "").strip() if legal_risk else ""

    try:
        urgency = normalize_urgency(raw.get("urgency", ""))
    except ValueError:
        urgency = "Alta" if legal_risk else ("Média" if sentiment == "Negativo" else "Baixa")
    if legal_risk:
        urgency = "Alta"

    summary = " ".join(str(raw.get("summary") or "").split()[:10]).strip(" .")
    if not summary:
        raise ValueError("'summary' não pode ser vazio")

    return {
        "sentiment": sentiment,
        "aspects": aspects,
        "urgency": urgency,
        "legal_risk": legal_risk,
        "legal_reason": legal_reason,
        "summary": summary,
    }
