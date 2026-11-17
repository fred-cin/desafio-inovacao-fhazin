"""DCL — Dashboard de Construtoras Locais.

Painel de inteligência de reputação para construtoras: lê avaliações de clientes,
classifica por aspecto com IA e transforma o resultado em decisões (onde agir,
o que responder primeiro, como estou contra a concorrência).
"""

from __future__ import annotations

import json
from datetime import timedelta

import altair as alt
import pandas as pd
import streamlit as st

from dcl import analyzer, insights, llm
from dcl.charts import aspect_chart
from dcl.data import (
    EVAL_PATH,
    TEMPLATE_PATH,
    explode_aspects,
    load_processed,
    read_reviews_csv,
    records_from_analysis,
    to_csv_bytes,
    to_frame,
)
from dcl.taxonomy import CATEGORIES

st.set_page_config(page_title="DCL · Reputação de Construtoras", page_icon="🏗️", layout="wide")

# ---------------------------------------------------------------- estilo
# Identidade "Frutiger Aero" do projeto original, mais contida: o vidro fica só nos cartões.
COMPANY_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SENT_COLORS = {"Positivo": "#2a78d6", "Neutro": "#b5b4ad", "Negativo": "#e34948"}
INK, INK_2 = "#0b3b52", "#4a6878"

st.markdown(
    f"""
    <style>
    .stApp {{ background: linear-gradient(180deg, #e3f4fb 0%, #eefaf1 45%, #fbfdfc 100%); }}
    h1, h2, h3 {{ color: {INK} !important; letter-spacing: -0.01em; }}
    [class*="st-key-card"] {{
        background: rgba(255,255,255,0.62);
        backdrop-filter: blur(10px); -webkit-backdrop-filter: blur(10px);
        border: 1px solid rgba(255,255,255,0.9) !important;
        border-radius: 18px !important;
        box-shadow: 0 6px 18px rgba(11,59,82,0.07), inset 0 1px 0 #fff;
        padding: 14px 16px;
    }}
    [data-testid="stMetricValue"] {{ color: {INK}; }}
    [data-testid="stSidebar"] {{ background: rgba(255,255,255,0.55); backdrop-filter: blur(8px); }}
    .dcl-quote {{ border-left: 3px solid #9ec5f4; padding: 2px 0 2px 12px; color: {INK}; margin: 6px 0 10px; }}
    .dcl-chip {{ display:inline-block; padding:1px 9px; margin:0 6px 4px 0; border-radius:999px;
                 font-size:0.8rem; background:#eef3f6; color:{INK_2}; border:1px solid #dbe6ec; }}
    .dcl-chip.neg {{ background:#fdeeee; color:#9a2424; border-color:#f6cfcf; }}
    .dcl-chip.pos {{ background:#e9f1fc; color:#1c5cab; border-color:#cde2fb; }}
    .dcl-sub {{ color:{INK_2}; margin-top:-10px; }}
    </style>
    """,
    unsafe_allow_html=True,
)


def chart_config(chart: alt.Chart) -> alt.Chart:
    return (
        chart.configure_view(strokeWidth=0)
        .configure(background="transparent")
        .configure_axis(labelColor=INK_2, titleColor=INK_2, gridColor="#e4ecef", domainColor="#cfdbe1", labelFontSize=12)
        .configure_legend(labelColor=INK_2, titleColor=INK_2, orient="bottom")
    )


def chips(aspects: list[dict]) -> str:
    cls = {"Negativo": "neg", "Positivo": "pos", "Neutro": ""}
    sign = {"Negativo": "−", "Positivo": "+", "Neutro": "·"}
    return "".join(f'<span class="dcl-chip {cls[a["sentiment"]]}">{sign[a["sentiment"]]} {a["category"]}</span>' for a in aspects)


# ---------------------------------------------------------------- dados
@st.cache_data
def load_base():
    return load_processed()


if "df" not in st.session_state:
    base_df, base_meta = load_base()
    st.session_state.df = base_df
    st.session_state.meta = base_meta
    st.session_state.cache = {}

df_all: pd.DataFrame = st.session_state.df
meta: dict = st.session_state.meta
companies = sorted(df_all["company"].unique())
color_scale = alt.Scale(domain=companies, range=COMPANY_COLORS[: len(companies)])
ai_on = llm.is_available()

# ---------------------------------------------------------------- barra lateral
with st.sidebar:
    st.markdown("### 🏗️ DCL")
    st.caption("Inteligência de reputação para construtoras")
    # Abre na empresa com mais avaliações urgentes: é onde o painel tem mais a mostrar.
    most_urgent = df_all[df_all["urgency"] == "Alta"]["company"].value_counts()
    default_company = most_urgent.index[0] if not most_urgent.empty else companies[0]
    company = st.selectbox("Sua construtora", companies, index=companies.index(default_company), key="company", help="As abas Visão geral e Alertas mostram esta empresa. As demais viram concorrentes.")
    dmin, dmax = df_all["date"].min().date(), df_all["date"].max().date()
    period = st.slider("Período", min_value=dmin, max_value=dmax, value=(dmin, dmax), format="DD/MM/YYYY")
    sources = st.multiselect("Fontes", sorted(df_all["source"].unique()), default=sorted(df_all["source"].unique()))
    st.divider()
    if ai_on:
        st.badge(f"IA conectada · {llm.get_model().split('/')[-1]}", icon=":material/bolt:", color="green")
    else:
        st.badge("Modo demonstração", icon=":material/science:", color="gray",
                 help="Sem chave de API configurada: a classificação usa regras locais e os textos usam modelos prontos.")
    engines = ", ".join(meta.get("engines", []))
    st.caption(f"Base classificada por: **{engines or '—'}**")

df = df_all[
    (df_all["date"].dt.date >= period[0]) & (df_all["date"].dt.date <= period[1]) & (df_all["source"].isin(sources))
]

# ---------------------------------------------------------------- cabeçalho
st.title("Reputação de construtoras, em tempo de decisão")
st.markdown(
    '<p class="dcl-sub">Avaliações de clientes classificadas por IA em aspectos — prazo, acabamento, assistência, '
    "financeiro — para mostrar onde agir primeiro.</p>",
    unsafe_allow_html=True,
)
if meta.get("demo_data", True):
    st.info(
        "**Dados de demonstração.** As construtoras são fictícias e as avaliações foram simuladas para este protótipo. "
        "Para usar dados reais, importe um CSV na aba *Analisar avaliações*.",
        icon=":material/info:",
    )

if df.empty:
    st.warning("Nenhuma avaliação no período e nas fontes selecionadas.")
    st.stop()

tab_overview, tab_bench, tab_alerts, tab_analyze, tab_how = st.tabs(
    ["Visão geral", "Comparativo", "Alertas e respostas", "Analisar avaliações", "Como funciona"]
)

# ================================================================ VISÃO GERAL
with tab_overview:
    mine = df[df["company"] == company]
    recent, previous = insights.period_split(mine)
    idx_r, idx_p = insights.reputation_index(recent["sentiment"]), insights.reputation_index(previous["sentiment"])

    k1, k2, k3, k4 = st.columns(4)
    with k1, st.container(key="card_k1"):
        st.metric("Índice de Reputação · 90 dias", "—" if idx_r is None else f"{idx_r:+d}",
                  None if idx_r is None or idx_p is None else f"{idx_r - idx_p:+d} vs. 90 dias anteriores",
                  help="% de avaliações positivas − % de negativas. Vai de −100 a +100, como um NPS.")
    with k2, st.container(key="card_k2"):
        st.metric("Avaliações no período", len(mine))
    with k3, st.container(key="card_k3"):
        st.metric("Negativas", f"{round(100 * (mine['sentiment'] == 'Negativo').mean()) if len(mine) else 0}%")
    with k4, st.container(key="card_k4"):
        st.metric("Pedem resposta urgente", int((mine["urgency"] == "Alta").sum()))
        st.caption(f"{int(mine['legal_risk'].sum())} com risco jurídico")

    left, right = st.columns([3, 2])
    with left, st.container(key="card_aspects"):
        st.subheader("O que os clientes elogiam e criticam")
        long = explode_aspects(mine)
        long = long[(long["aspect_sentiment"] != "Neutro") & (long["category"] != "Outro")]
        if long.empty:
            st.caption("Sem menções suficientes no período.")
        else:
            counts = long.groupby(["category", "aspect_sentiment"]).size().reset_index(name="n")
            counts["value"] = counts.apply(lambda r: -r["n"] if r["aspect_sentiment"] == "Negativo" else r["n"], axis=1)
            order = (counts[counts["aspect_sentiment"] == "Negativo"].sort_values("n", ascending=False)["category"].tolist()
                     + [c for c in CATEGORIES if c not in set(counts[counts["aspect_sentiment"] == "Negativo"]["category"])])
            st.altair_chart(chart_config(aspect_chart(counts, order)), width="stretch")
            st.caption("Uma avaliação pode citar mais de um aspecto — por isso a análise é por aspecto, não por avaliação.")

    with right, st.container(key="card_trend"):
        st.subheader("Evolução por trimestre")
        trend = insights.quarterly_trend(df)
        market = (trend.assign(w=trend["index"] * trend["n"]).groupby("quarter_start", as_index=False)
                  .agg(w=("w", "sum"), n=("n", "sum")))
        market["index"] = market["w"] / market["n"]
        mine_t = trend[trend["company"] == company]
        line = alt.Chart(mine_t).mark_line(point=alt.OverlayMarkDef(size=70, filled=True), strokeWidth=2.5,
                                            color=COMPANY_COLORS[companies.index(company)]).encode(
            x=alt.X("quarter_start:T", title=None, axis=alt.Axis(format="%m/%Y", tickCount="month")),
            y=alt.Y("index:Q", title="Índice de Reputação", scale=alt.Scale(domain=[-100, 100])),
            tooltip=[alt.Tooltip("quarter_label:N", title="Trimestre"), alt.Tooltip("index:Q", title="Índice"),
                     alt.Tooltip("n:Q", title="Avaliações")],
        )
        mkt = alt.Chart(market).mark_line(strokeDash=[5, 4], strokeWidth=2, color="#9fb3bd").encode(
            x="quarter_start:T", y="index:Q", tooltip=[alt.Tooltip("index:Q", title="Média do mercado", format=".0f")])
        zero = alt.Chart(pd.DataFrame({"y": [0]})).mark_rule(color="#cfdbe1").encode(y="y:Q")
        st.altair_chart(chart_config(alt.layer(zero, mkt, line).properties(height=260)), width="stretch")
        st.caption(f"Linha cheia: {company}. Tracejada: média das construtoras monitoradas.")

    with st.container(key="card_summary"):
        st.subheader("Resumo executivo")
        cache_key = ("summary", company, period, tuple(sources), len(df_all))
        text, engine = st.session_state.cache.get(cache_key) or insights.executive_summary(df, company, use_llm=False)
        st.markdown(text)
        c1, c2 = st.columns([1, 3])
        with c1:
            if st.button("Gerar análise com IA", icon=":material/auto_awesome:", disabled=not ai_on,
                         help=None if ai_on else "Disponível quando a chave de API está configurada."):
                with st.spinner("Escrevendo o resumo..."):
                    st.session_state.cache[cache_key] = insights.executive_summary(df, company, use_llm=True)
                st.rerun()
        with c2:
            st.caption(f"Gerado por: {engine}. Pensado para o e-mail semanal da diretoria.")

# ================================================================ COMPARATIVO
with tab_bench:
    board = insights.company_scoreboard(df)
    with st.container(key="card_board"):
        st.subheader("Ranking de reputação")
        st.dataframe(
            board.style.apply(lambda r: ["font-weight:700; background-color:#e9f1fc" if r["Construtora"] == company else "" for _ in r], axis=1),
            hide_index=True, width="stretch",
            column_config={
                "Índice de Reputação": st.column_config.NumberColumn(format="%+d", help="Período inteiro selecionado"),
                "Últimos 90 dias": st.column_config.NumberColumn(format="%+d"),
                "Variação": st.column_config.NumberColumn(format="%+d", help="Últimos 90 dias vs. 90 dias anteriores"),
                "% negativas": st.column_config.NumberColumn(format="%d%%"),
            },
        )

    matrix = insights.category_matrix(df)
    with st.container(key="card_heat"):
        st.subheader("Onde cada construtora ganha e perde")
        heat = alt.Chart(matrix).mark_rect(cornerRadius=4, stroke="#fbfdfc", strokeWidth=2).encode(
            x=alt.X("category:N", title=None, sort=list(CATEGORIES), axis=alt.Axis(labelAngle=0, labelLimit=110,
                    labelExpr="split(datum.label, ' e ')[0]")),
            y=alt.Y("company:N", title=None, sort=companies),
            color=alt.Color("net:Q", title="Saldo", scale=alt.Scale(domain=[-100, 0, 100], range=["#e34948", "#f0efec", "#2a78d6"])),
            tooltip=[alt.Tooltip("company:N", title="Construtora"), alt.Tooltip("category:N", title="Categoria"),
                     alt.Tooltip("net:Q", title="Saldo (+pos −neg)", format="+.0f"), alt.Tooltip("mentions:Q", title="Menções")],
        )
        labels = alt.Chart(matrix).mark_text(fontSize=12, fontWeight=600).encode(
            x=alt.X("category:N", sort=list(CATEGORIES)), y=alt.Y("company:N", sort=companies),
            text=alt.Text("net:Q", format="+.0f"),
            color=alt.condition("abs(datum.net) > 60", alt.value("#ffffff"), alt.value(INK)),
        )
        st.altair_chart(chart_config(alt.layer(heat, labels).properties(height=60 * len(companies) + 40)), width="stretch")
        st.caption("Saldo = (menções positivas − negativas) ÷ menções da categoria. Passe o mouse para ver o volume.")

        # Comparação direta: onde a empresa selecionada fica mais atrás do melhor concorrente.
        pivot = matrix.pivot(index="category", columns="company", values="net")
        if company in pivot and pivot.shape[1] > 1:
            others = pivot.drop(columns=company)
            gap = (others.max(axis=1) - pivot[company]).dropna().sort_values(ascending=False)
            gap = gap[gap > 0].head(3)
            if not gap.empty:
                st.markdown("**Maiores distâncias para o melhor concorrente:**")
                for cat, g in gap.items():
                    leader = others.loc[cat].idxmax()
                    st.markdown(f"- **{cat}**: {company} {pivot.loc[cat, company]:+.0f} vs. {leader} {others.loc[cat].max():+.0f} ({g:.0f} pontos)")

    with st.container(key="card_trend_all"):
        st.subheader("Evolução do índice")
        trend = insights.quarterly_trend(df)
        lines = alt.Chart(trend).mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2).encode(
            x=alt.X("quarter_start:T", title=None, axis=alt.Axis(format="%m/%Y", tickCount="month")),
            y=alt.Y("index:Q", title="Índice de Reputação", scale=alt.Scale(domain=[-100, 100])),
            color=alt.Color("company:N", title=None, scale=color_scale),
            tooltip=[alt.Tooltip("company:N", title="Construtora"), alt.Tooltip("quarter_label:N", title="Trimestre"),
                     alt.Tooltip("index:Q", title="Índice"), alt.Tooltip("n:Q", title="Avaliações")],
        )
        st.altair_chart(chart_config(lines.properties(height=280)), width="stretch")

# ================================================================ ALERTAS E RESPOSTAS
with tab_alerts:
    queue = insights.alerts(df, company)
    f1, f2 = st.columns([2, 1])
    with f1:
        level = st.segmented_control("Urgência", ["Alta", "Média"], selection_mode="multi", default=["Alta", "Média"])
    with f2:
        only_legal = st.toggle("Só risco jurídico")
    queue = queue[queue["urgency"].isin(level or [])]
    if only_legal:
        queue = queue[queue["legal_risk"]]

    st.caption(f"{len(queue)} avaliações na fila de {company}, da mais urgente para a menos urgente. "
               "Respostas rápidas no Reclame Aqui melhoram a nota pública da empresa.")
    for _, row in queue.head(20).iterrows():
        with st.container(key=f"card_alert_{row['id']}"):
            h1, h2 = st.columns([4, 1])
            with h1:
                badge_color = "red" if row["urgency"] == "Alta" else "orange"
                st.badge(f"Urgência {row['urgency'].lower()}", color=badge_color)
                if row["legal_risk"]:
                    st.badge("Risco jurídico", icon=":material/gavel:", color="violet", help=row.get("legal_reason") or None)
            with h2:
                st.caption(f"{row['source']} · {row['date']:%d/%m/%Y}")
            st.markdown(f'<div class="dcl-quote">{row["text"]}</div>{chips(row["aspects"])}', unsafe_allow_html=True)

            reply_key = f"reply_{row['id']}"
            if st.button("Sugerir resposta", key=f"btn_{row['id']}", icon=":material/edit_note:"):
                with st.spinner("Escrevendo..."):
                    st.session_state[reply_key], st.session_state[f"{reply_key}_eng"] = insights.draft_reply(row.to_dict(), company)
            if reply_key in st.session_state:
                st.text_area("Rascunho (edite antes de publicar)", key=reply_key, height=120)
                st.caption(f"Gerado por: {st.session_state[f'{reply_key}_eng']}. Sempre revisado por uma pessoa antes de publicar.")

    if not queue.empty:
        st.download_button("Baixar fila em CSV", to_csv_bytes(queue), file_name=f"alertas_{company}.csv", mime="text/csv",
                           icon=":material/download:")

# ================================================================ ANALISAR AVALIAÇÕES
with tab_analyze:
    st.subheader("Analise avaliações novas")
    st.caption("Cole avaliações ou importe uma planilha. A IA classifica cada uma e você pode incluir o resultado no painel.")
    max_batch = int(llm.get_setting("DCL_MAX_BATCH", "50"))
    mode = st.radio("Entrada", ["Colar texto", "Importar CSV"], horizontal=True, label_visibility="collapsed")

    raw = None
    if mode == "Colar texto":
        c1, c2 = st.columns(2)
        with c1:
            new_company = st.text_input("Construtora", value=company)
        with c2:
            new_source = st.selectbox("Fonte", ["Google Maps", "Reclame Aqui", "Instagram", "Formulário pós-entrega", "Outra"])
        pasted = st.text_area(
            "Uma avaliação por linha", height=160,
            placeholder="Ex.: Entregaram com 3 meses de atraso, mas o acabamento ficou ótimo.\nAssistência técnica não responde há duas semanas.",
        )
        lines = [l.strip() for l in pasted.splitlines() if l.strip()]
        if lines:
            raw = pd.DataFrame({"id": [f"novo-{i:03d}" for i in range(1, len(lines) + 1)], "company": new_company,
                                "source": new_source, "date": pd.Timestamp.today().date(), "text": lines})
    else:
        c1, c2 = st.columns([3, 1])
        with c1:
            up = st.file_uploader("CSV com uma coluna de texto (opcional: empresa, fonte, data)", type=["csv"])
        with c2:
            st.download_button("Modelo de planilha", TEMPLATE_PATH.read_bytes(), file_name="modelo_importacao.csv",
                               mime="text/csv", icon=":material/table:")
        if up is not None:
            try:
                raw = read_reviews_csv(up.getvalue(), default_company=company)
                st.caption(f"{len(raw)} avaliações encontradas.")
            except Exception as exc:  # mensagem amigável para planilha fora do padrão
                st.error(f"Não consegui ler o arquivo: {exc}")

    if raw is not None and len(raw) > max_batch:
        st.warning(f"Neste deploy público o limite é {max_batch} avaliações por vez; vou analisar as primeiras {max_batch}.")
        raw = raw.head(max_batch)

    if st.button("Analisar", type="primary", icon=":material/auto_awesome:", disabled=raw is None):
        bar = st.progress(0.0, text="Analisando...")
        analyses = analyzer.analyze_many(raw["text"], on_progress=lambda i, n: bar.progress(i / n, text=f"Analisando {i}/{n}"))
        bar.empty()
        st.session_state.new_results = to_frame(records_from_analysis(raw, analyses))

    res = st.session_state.get("new_results")
    if res is not None:
        with st.container(key="card_results"):
            for _, r in res.iterrows():
                tag = {"Positivo": "🔵", "Neutro": "⚪", "Negativo": "🔴"}[r["sentiment"]]
                extra = " · ⚖️ risco jurídico" if r["legal_risk"] else ""
                st.markdown(f"{tag} **{r['summary']}** · urgência {r['urgency'].lower()}{extra}<br>{chips(r['aspects'])}",
                            unsafe_allow_html=True)
            engines_used = ", ".join(sorted(res["engine"].unique()))
            st.caption(f"Classificado por: {engines_used}")
            b1, b2, _ = st.columns([1, 1, 2])
            with b1:
                if st.button("Incluir no painel", icon=":material/add_chart:"):
                    st.session_state.df = pd.concat([st.session_state.df, res], ignore_index=True)
                    st.session_state.meta = {**meta, "engines": sorted(set(meta.get("engines", [])) | set(res["engine"]))}
                    st.session_state.pop("new_results")
                    st.toast("Avaliações incluídas. Os gráficos já foram atualizados.")
                    st.rerun()
            with b2:
                st.download_button("Baixar CSV", to_csv_bytes(res), file_name="avaliacoes_classificadas.csv", mime="text/csv")

# ================================================================ COMO FUNCIONA
with tab_how:
    st.subheader("Como funciona")
    st.markdown(
        """
1. **Coleta** — avaliações do Google Maps, Reclame Aqui, Instagram e formulários pós-entrega, importadas por CSV.
2. **Classificação por aspecto** — um modelo de linguagem lê cada avaliação e devolve JSON validado: sentimento geral,
   aspectos citados (cada um com o próprio sentimento), urgência, risco jurídico e um resumo factual.
   Respostas fora do formato são rejeitadas e o modelo recebe o erro para corrigir (até 3 tentativas).
3. **Métricas** — Índice de Reputação (lógica de NPS), saldo por categoria, comparação com concorrentes e tendência.
4. **Ação** — fila de alertas por urgência, rascunhos de resposta e resumo executivo para a diretoria.

Sem chave de API, o painel continua funcionando com regras locais (*modo demonstração*).
        """
    )
    st.markdown("**Categorias**")
    st.dataframe(pd.DataFrame({"Categoria": list(CATEGORIES), "O que entra": list(CATEGORIES.values())}),
                 hide_index=True, width="stretch")

    st.markdown("**Quão bem a classificação acerta**")
    if EVAL_PATH.exists():
        evals = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
        table = pd.DataFrame([{
            "Motor": e["engine"],
            "Avaliações": e["n"],
            "Sentimento (acurácia)": f"{100 * e['sentiment_accuracy']:.0f}%",
            "Categorias (F1)": f"{100 * e['category_micro']['f1']:.0f}%",
            "Aspecto completo (F1)": f"{100 * e['aspect_micro']['f1']:.0f}%",
            "Risco jurídico (recall)": f"{100 * e['legal_risk']['recall']:.0f}%",
        } for e in evals])
        st.dataframe(table, hide_index=True, width="stretch")
        st.caption("Comparação com rótulos de referência feitos à mão (`data/rotulos_referencia.csv`, gerado por `evaluate.py`). "
                   "As regras locais foram ajustadas neste mesmo conjunto, então o número delas é otimista: "
                   "servem de linha de base, não de prova.")
    else:
        st.caption("Rode `python evaluate.py` para gerar as métricas.")
