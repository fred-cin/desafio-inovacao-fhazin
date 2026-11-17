"""Métricas de reputação, resumo executivo e rascunho de respostas."""

from __future__ import annotations

import json

import pandas as pd

from . import llm
from .data import explode_aspects

URGENCY_ORDER = {"Alta": 0, "Média": 1, "Baixa": 2}


def reputation_index(sentiments: pd.Series) -> float | None:
    """Índice de Reputação: % positivas − % negativas, de −100 a +100 (lógica de NPS)."""
    n = len(sentiments)
    if n == 0:
        return None
    return round(100 * ((sentiments == "Positivo").sum() - (sentiments == "Negativo").sum()) / n)


def period_split(df: pd.DataFrame, days: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Últimos `days` dias vs. os `days` dias anteriores, a partir da avaliação mais recente."""
    if df.empty:
        return df, df
    end = df["date"].max()
    recent = df[df["date"] > end - pd.Timedelta(days=days)]
    previous = df[(df["date"] <= end - pd.Timedelta(days=days)) & (df["date"] > end - pd.Timedelta(days=2 * days))]
    return recent, previous


def company_scoreboard(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for company, g in df.groupby("company"):
        recent, previous = period_split(g)
        ir, ip = reputation_index(recent["sentiment"]), reputation_index(previous["sentiment"])
        rows.append({
            "Construtora": company,
            "Avaliações": len(g),
            "Índice de Reputação": reputation_index(g["sentiment"]),
            "Últimos 90 dias": ir,
            "Variação": None if ir is None or ip is None else ir - ip,
            "% negativas": round(100 * (g["sentiment"] == "Negativo").mean()),
            "Riscos jurídicos": int(g["legal_risk"].sum()),
        })
    return pd.DataFrame(rows).sort_values("Índice de Reputação", ascending=False).reset_index(drop=True)


def category_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Saldo (positivas − negativas) / menções por construtora × categoria."""
    long = explode_aspects(df)
    long = long[long["category"] != "Outro"]
    grp = long.groupby(["company", "category"])["aspect_sentiment"]
    out = grp.agg(
        mentions="size",
        positives=lambda s: (s == "Positivo").sum(),
        negatives=lambda s: (s == "Negativo").sum(),
    ).reset_index()
    out["net"] = (100 * (out["positives"] - out["negatives"]) / out["mentions"]).round()
    return out


def quarterly_trend(df: pd.DataFrame) -> pd.DataFrame:
    q = df.assign(quarter=df["date"].dt.to_period("Q"))
    out = q.groupby(["company", "quarter"])["sentiment"].agg(index=reputation_index, n="size").reset_index()
    out["quarter_start"] = out["quarter"].dt.start_time
    out["quarter_label"] = out["quarter"].apply(lambda p: f"{p.quarter}º tri {p.year}")
    return out


def top_issues(df: pd.DataFrame, company: str, days: int = 90) -> pd.DataFrame:
    """Categorias com mais menções negativas da empresa, com variação vs. período anterior."""
    g = df[df["company"] == company]
    recent, previous = period_split(g, days)

    def neg_counts(part: pd.DataFrame) -> pd.Series:
        if part.empty:
            return pd.Series(dtype=int)
        long = explode_aspects(part)
        return long[long["aspect_sentiment"] == "Negativo"].groupby("category").size()

    r, p = neg_counts(recent), neg_counts(previous)
    out = pd.DataFrame({"recentes": r, "anteriores": p}).fillna(0).astype(int)
    out["variação"] = out["recentes"] - out["anteriores"]
    out.index.name = "category"
    return out.sort_values(["recentes", "variação"], ascending=False)


def alerts(df: pd.DataFrame, company: str) -> pd.DataFrame:
    g = df[(df["company"] == company) & (df["sentiment"] != "Positivo") & (df["urgency"] != "Baixa")].copy()
    g["_u"] = g["urgency"].map(URGENCY_ORDER)
    return g.sort_values(["_u", "legal_risk", "date"], ascending=[True, False, False]).drop(columns="_u")


def _facts(df: pd.DataFrame, company: str) -> dict:
    g = df[df["company"] == company]
    recent, previous = period_split(g)
    issues = top_issues(df, company)
    board = company_scoreboard(df)
    rank = int(board.index[board["Construtora"] == company][0]) + 1 if company in set(board["Construtora"]) else None
    positives = explode_aspects(g)
    positives = positives[positives["aspect_sentiment"] == "Positivo"].groupby("category").size().sort_values(ascending=False)
    return {
        "empresa": company,
        "avaliacoes_total": len(g),
        "indice_reputacao_total": reputation_index(g["sentiment"]),
        "indice_ultimos_90_dias": reputation_index(recent["sentiment"]),
        "indice_90_dias_anteriores": reputation_index(previous["sentiment"]),
        "posicao_entre_concorrentes": f"{rank}º de {len(board)}" if rank else None,
        "principais_problemas_90_dias": issues.head(3).reset_index().to_dict("records"),
        "pontos_fortes": positives.head(2).to_dict(),
        "alertas_alta_urgencia": int((g["urgency"] == "Alta").sum()),
        "riscos_juridicos": int(g["legal_risk"].sum()),
        "exemplos_recentes_negativos": recent[recent["sentiment"] == "Negativo"]["text"].head(4).tolist(),
    }


def _fmt_delta(a, b) -> str:
    if a is None or b is None:
        return "sem base de comparação"
    d = a - b
    return f"{'+' if d > 0 else ''}{d} pontos vs. trimestre anterior"


def executive_summary(df: pd.DataFrame, company: str, use_llm: bool | None = None) -> tuple[str, str]:
    """Devolve (markdown, motor). Com IA: análise escrita; sem IA: relatório montado das métricas."""
    facts = _facts(df, company)
    if use_llm is None:
        use_llm = llm.is_available()
    if use_llm:
        system = (
            "Você é consultor de reputação de construtoras. Escreva um resumo executivo em português do Brasil, "
            "em Markdown, para a diretoria. No máximo 5 tópicos curtos e 1 recomendação final em negrito. "
            "Use apenas os números fornecidos, não invente dados. Seja direto e específico."
        )
        try:
            return llm.chat_text(system, json.dumps(facts, ensure_ascii=False, default=str)), f"ia:{llm.get_model()}"
        except Exception:
            pass

    lines = [
        f"- **Índice de Reputação (90 dias): {facts['indice_ultimos_90_dias']}** "
        f"({_fmt_delta(facts['indice_ultimos_90_dias'], facts['indice_90_dias_anteriores'])}).",
    ]
    if facts["posicao_entre_concorrentes"]:
        lines.append(f"- Posição entre as construtoras monitoradas: **{facts['posicao_entre_concorrentes']}**.")
    if facts["principais_problemas_90_dias"]:
        top = facts["principais_problemas_90_dias"][0]
        lines.append(
            f"- Maior fonte de reclamações recentes: **{top['category']}** "
            f"({top['recentes']} menções negativas, {'+' if top['variação'] > 0 else ''}{top['variação']} vs. período anterior)."
        )
    if facts["pontos_fortes"]:
        lines.append(f"- Pontos fortes mais citados: {', '.join(facts['pontos_fortes'])}.")
    lines.append(
        f"- {facts['alertas_alta_urgencia']} avaliações de alta urgência e "
        f"{facts['riscos_juridicos']} com risco jurídico pedem resposta prioritária."
    )
    return "\n".join(lines), "métricas (sem IA)"


REPLY_TEMPLATES = {
    "Prazo de entrega": "Entendemos a frustração com o prazo e pedimos desculpas pelo transtorno. Queremos te passar a previsão atualizada e o andamento da obra pessoalmente.",
    "Acabamento e qualidade": "Lamentamos que o imóvel não esteja no padrão que você esperava. Nossa equipe técnica vai avaliar o problema.",
    "Assistência técnica": "Pedimos desculpas pela demora no atendimento do seu chamado. Já encaminhamos o caso para a coordenação da assistência técnica, com prioridade.",
    "Atendimento e vendas": "Sentimos muito pela falta de retorno. Você merece respostas claras e rápidas.",
    "Financeiro e contrato": "Entendemos sua preocupação com os valores. Queremos revisar o seu caso com cuidado, item por item.",
    "Área comum e infraestrutura": "Obrigado por nos avisar. Vamos verificar a situação junto à administração do condomínio.",
}


def draft_reply(review: dict, company: str, use_llm: bool | None = None) -> tuple[str, str]:
    """Rascunho de resposta pública à avaliação. Sempre revisado por uma pessoa antes de publicar."""
    if use_llm is None:
        use_llm = llm.is_available()
    if use_llm:
        system = (
            f"Você escreve respostas públicas da {company} a avaliações de clientes (Reclame Aqui, Google). "
            "Português do Brasil, tom humano e respeitoso, no máximo 70 palavras. Reconheça o problema específico, "
            "não culpe o cliente, não prometa indenização nem prazos que não foram informados, convide para um canal "
            "privado (relacionamento@construtora.com.br). Se houver menção a Procon, advogado ou Justiça, seja ainda "
            "mais cuidadoso e não admita culpa jurídica. Responda só com o texto da resposta."
        )
        try:
            return llm.chat_text(system, f"Avaliação: {review['text']}", max_tokens=250), f"ia:{llm.get_model()}"
        except Exception:
            pass

    negatives = [a["category"] for a in review["aspects"] if a["sentiment"] == "Negativo"]
    body = REPLY_TEMPLATES.get(negatives[0] if negatives else "", "Agradecemos o seu retorno, ele é importante para nós.")
    reply = (
        f"Olá! Aqui é a equipe de relacionamento da {company}. {body} "
        "Pode nos chamar em relacionamento@construtora.com.br com o número da sua unidade? "
        "Vamos acompanhar seu caso até a solução."
    )
    return reply, "modelo pronto (sem IA)"
