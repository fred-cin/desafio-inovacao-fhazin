"""Gráficos reutilizáveis (separados do app para poderem ser renderizados e testados sozinhos)."""

from __future__ import annotations

import altair as alt
import pandas as pd

SENT_COLORS = {"Positivo": "#2a78d6", "Negativo": "#e34948"}


def aspect_chart(counts: pd.DataFrame, order: list[str]) -> alt.LayerChart:
    """Barras divergentes: menções negativas à esquerda, positivas à direita, uma linha por categoria.

    `counts` tem as colunas category, aspect_sentiment ('Negativo'/'Positivo'), n e value (n negativo para Negativo).
    Todos os nomes de categoria aparecem, inteiros: sem rótulo, a barra não diz o que mede.
    """
    present = [c for c in order if c in set(counts["category"])]
    bars = alt.Chart(counts).mark_bar(cornerRadius=4, height=20).encode(
        x=alt.X(
            "value:Q",
            title="← menções negativas · menções positivas →",
            axis=alt.Axis(format="d", labelExpr="abs(datum.value)", tickMinStep=1),
        ),
        y=alt.Y(
            "category:N",
            sort=present,
            title=None,
            axis=alt.Axis(labelLimit=240, labelOverlap=False, labelPadding=8, labelFontSize=12),
        ),
        color=alt.Color(
            "aspect_sentiment:N",
            title=None,
            scale=alt.Scale(domain=["Negativo", "Positivo"], range=[SENT_COLORS["Negativo"], SENT_COLORS["Positivo"]]),
        ),
        tooltip=[
            alt.Tooltip("category:N", title="Categoria"),
            alt.Tooltip("aspect_sentiment:N", title="Sentimento"),
            alt.Tooltip("n:Q", title="Menções"),
        ],
    )
    rule = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#9fb3bd").encode(x="x:Q")
    return alt.layer(bars, rule).properties(height=max(220, 40 * len(present) + 50))
