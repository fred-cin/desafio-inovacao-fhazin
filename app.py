import streamlit as st
import pandas as pd
import json

# construir a interface do dashboard
st.set_page_config(page_title="Dashboard de Construtoras Locais (DCL)", layout="wide")

#design baseado na paleta de cores Frutiger Aero
st.markdown("""
    <style>
    /* Frutiger Aero */
    .stApp {
        background: linear-gradient(180deg, #d4f1f9 0%, #e5f9e0 50%, #ffffff 100%);
    }
    
    /* azul */
    h1, h2, h3, p {
        color: #004c6d !important;
        text-shadow: 1px 1px 2px rgba(255, 255, 255, 0.8);
    }
    
    /* Aero Glass */
    [data-testid="stVerticalBlock"] > div {
        background: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05), inset 0 1px 0 rgba(255, 255, 255, 1);
        padding: 10px;
    }
    
    /* divisores */
    hr {
        border: 0;
        height: 2px;
        background-image: linear-gradient(to right, rgba(0, 153, 255, 0), rgba(0, 204, 153, 0.75), rgba(0, 153, 255, 0));
    }
    </style>
""", unsafe_allow_html=True)

st.title("DCL: Dashboard de Construtoras Locais")
st.write("Inteligência de mercado automatizada para construtoras locais.")

# carregar os dados processados
with open("processed_reviews.json", "r", encoding="utf-8") as file:
    data = json.load(file)

df = pd.DataFrame(data)
df['sentiment'] = df['sentiment'].replace({'Negative': 'Negativo', 'Positive': 'Positivo', 'Neutral': 'Neutro'})

# criar gráficos de análise de sentimentos e categorias
col1, col2 = st.columns(2)

with col1:
    st.subheader("Análise de Sentimentos do Cliente")
    sentiment_counts = df['sentiment'].value_counts()
    st.bar_chart(sentiment_counts, color="#00cc99")

with col2:
    st.subheader("Categorias de Reclamações")
    category_counts = df['category'].value_counts()
    st.bar_chart(category_counts, color="#0099ff")

st.divider()

# mostrar a tabela de dados processados
st.subheader("Banco de Dados de Avaliações Recentes")
st.dataframe(df[["sentiment", "category", "summary", "original_text"]], use_container_width=True)