"""Classificador local por regras: o "modo demonstração".

Não substitui a IA. Existe para que o painel funcione sem chave de API (avaliadores,
deploy sem segredo configurado, queda do provedor) e como linha de base na
avaliação: se o modelo não supera estas regras, não vale o custo.
"""

from __future__ import annotations

import re

from .taxonomy import fold, validate_analysis

# Raízes já sem acento (o texto é comparado depois de fold()).
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Prazo de entrega": ["atras", "prazo", "adia", "cronograma", "obra parada", "nem reboco",
                         "entregaram antes", "entregue no prazo", "valeu a espera", "dia marcado"],
    "Acabamento e qualidade": ["acabamento", "piso", "porcelanato", "esquadria", "janela", "porta ",
                               "rachadura", "fissura", "infiltra", "revestimento", "material",
                               "qualidade", "planta", "projeto", "construcao solida", "decorado",
                               "pendencia", "empenad", "desalinhad", "curto", "fechadura"],
    "Assistência técnica": ["assistencia", "chamado", "vistoria", "reparo", "garantia", "tecnico",
                            "consertar", "pos-obra", "visita"],
    "Atendimento e vendas": ["atendimento", "atendid", "corretor", "stand", "sac", "responde",
                             "retorno", "comunicacao", "telefone", "whatsapp", "direct", "e-mail",
                             "explicou", "explicar", "informacao", "atencios", "pos-venda", "ligue"],
    "Financeiro e contrato": ["taxa", "contrato", "distrato", "juros", "boleto", "parcela", "financeiro",
                              "reajuste", "incc", "valor", "devolu", "devolver", "cobra", "preco",
                              "pagamento", "custo-beneficio", "aluguel", "compensacao", "reter"],
    "Área comum e infraestrutura": ["lazer", "piscina", "academia", "condominio", "elevador", "garagem",
                                    "playground", "portaria", "paisagismo", "area comum", "predio",
                                    "subsolo", "fachada", "verde"],
    "Localização": ["localiza", "bairro", "perto", "proximo", "metro", "praia", "shopping", "afastad",
                    "escolas", "vizinhanca"],
}

POSITIVE = ["excelente", "otim", "incrivel", "impecavel", "perfeit", "satisfeit", "recomendo", "parabens",
            "nota 10", "gostei", "amei", "ama ", "feliz", "lind", "no prazo", "antes do prazo",
            "dentro do prazo", "dentro do cronograma", "rapid", "atencios", "eficiente", "de primeira",
            "alto padrao", "transparencia", "bem feit", "bem cuidad", "bem localizad", "bem distribuid",
            "bem atendid", "bem pensad", "ponto forte", "tranquil", "solida", "top", "vale cada",
            "valeu", "compensa", "flexive", "sem surpresa", "tudo certo", "boa qualidade", "de qualidade",
            "outro nivel", "caprichando", "modern", "organizad", "claro", "sem nenhuma pendencia",
            "resolvid", "resolveu", "bom ", "boa ", "bons", "funcionando bem", "ampla", "muito verde"]
NEGATIVE = ["pessim", "horrivel", "atraso", "atrasad", "adiad", "adiamento", "abusiv", "nao responde",
            "ninguem", "descaso", "decepcion", "ruim", "demora", "demais", "novela", "sufoco", "sumiu",
            "enganad", "errad", "recusa", "se nega", "impossivel", "nenhum", "nenhuma", "sem resposta",
            "sem solucao", "infiltra", "rachadura", "vazamento", "curto", "soltando", "empenad",
            "desalinhad", "apertad", "afastad", "salgad", "acima do", "deixa a desejar", "parad",
            "precisa melhorar", "caiu muito", "perdi", "empurra", "confusa", "perigoso", "sem aviso",
            "nao estava claro", "nada disso", "nao comparece", "nao veio", "nao fala", "so desculpas",
            "e-mail padrao", "pendencias", "problema", "esperando", "estou ha"]
NEUTRAL_HINTS = ["nada de excepcional", "nada de ruim"]

LEGAL_TERMS = ["procon", "advogad", "justica", "processar", "judicial", "juizado", "indeniza",
               "danos morais", "notific", "meus direitos", "distrato", "ministerio publico"]
SAFETY_TERMS = ["infiltra", "vazamento", "rachadura", "fissura", "curto", "eletric", "gas",
                "elevador", "soltando", "cair", "perigos", "seguranca", "desab"]

_CLAUSE_SPLIT = re.compile(r"[.!?;\n]+|,\s*(?:mas|porem|so que|apesar)\b|\b(?:mas|porem|entretanto|so que|apesar)\b")


def _count(text: str, terms: list[str]) -> int:
    return sum(text.count(t) for t in terms)


def _polarity(text: str) -> int:
    if any(h in text for h in NEUTRAL_HINTS):
        return 0
    pos, neg = _count(text, POSITIVE), _count(text, NEGATIVE)
    return (pos > neg) - (neg > pos)


# "nada de rachadura ou infiltração" é elogio: remove o trecho negado antes de contar termos.
_NEGATED = re.compile(r"\bnada de \w+(?:\s+(?:ou|e|nem)\s+\w+)?")


def analyze(text: str) -> dict:
    t = _NEGATED.sub(" ", fold(text)) + " "
    clauses = [c.strip() + " " for c in _CLAUSE_SPLIT.split(t) if c and c.strip()]

    overall = _polarity(t)
    aspects: dict[str, int] = {}
    for clause in clauses:
        cats = [c for c, kws in CATEGORY_KEYWORDS.items() if any(k in clause for k in kws)]
        if not cats:
            continue
        pol = _polarity(clause)
        for cat in cats:
            aspects[cat] = aspects.get(cat, 0) + (pol if pol else overall)

    has_pos = any(v > 0 for v in aspects.values())
    has_neg = any(v < 0 for v in aspects.values())
    if has_pos and has_neg and abs(_count(t, POSITIVE) - _count(t, NEGATIVE)) <= 1:
        overall = 0
    sign = {1: "Positivo", 0: "Neutro", -1: "Negativo"}
    sentiment = sign[overall]

    legal_terms = [k for k in LEGAL_TERMS if k in t]
    legal = bool(legal_terms)
    if legal:
        urgency = "Alta"
    elif sentiment != "Positivo" and any(k in t for k in SAFETY_TERMS):
        urgency = "Alta"
    elif sentiment == "Negativo":
        urgency = "Média"
    else:
        urgency = "Baixa"

    first = re.split(r"[.!?;\n]", str(text).strip())[0]
    words = first.split()
    summary = " ".join(words[:7]) + ("…" if len(words) > 7 else "")

    raw = {
        "sentiment": sentiment,
        "aspects": [{"category": c, "sentiment": sign[(v > 0) - (v < 0)]} for c, v in aspects.items()],
        "urgency": urgency,
        "legal_risk": legal,
        "legal_reason": f"menciona: {', '.join(legal_terms)}" if legal else "",
        "summary": summary or "avaliação sem texto",
    }
    return validate_analysis(raw)
