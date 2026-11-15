# desafio-inovacao-fhazin
Submissão para o desafio de inovação do PS do CITi de 2026.2.
# DCL: Dashboard de Construtoras Locais

**Inteligência de mercado automatizada (B2B SaaS) desenvolvida para o Desafio de Inovação CITi.**

🔗 **[Acesse aqui o Dashboard](https://desafio-inovacao-fhazin.streamlit.app/)**

## O Problema vs a Solução
Construtoras no Nordeste gerenciam obras de milhões, mas muitas vezes controlam sua reputação online e o feedback dos clientes de forma manual e ineficiente. A perda de reputação no mercado imobiliário afeta diretamente as vendas e a captação de novos clientes. 

O **DCL (Dashboard de Construtoras Locais)** resolve isso centralizando e automatizando a inteligência de mercado. A aplicação coleta avaliações brutas e utiliza Inteligência Artificial para extrair sentimentos e categorizar reclamações estruturais (ex: Atraso de Obra, Acabamento, Financeiro/Contrato), permitindo tomadas de decisão rápidas e baseadas em dados.

O MVP aqui mostrado é extremamente conveniente para ser convertido em ferramenta de outros setores empresariais (restaurantes, serviços de atendimento ao cliente, ou o próprio CITi), a questão das construtoras é sommente um exemplo.

## Arquitetura
O projeto evita a interface de "chat com IA" e opera como um pipeline de dados real, dividido em duas camadas principais:

1. **Pipeline de Processamento (Back-end AI):**
   * **Python & OpenRouter API:** O script `process_data.py` atua como um ETL automatizado. Ele consome dados de texto bruto (`raw_reviews.json`) e os envia para o modelo **Llama 3** (via OpenRouter).
   * **Sanitização de Dados:** O modelo atua sob um prompt estrito de sistema para ignorar ruídos e devolver a saída *exclusivamente* em um formato JSON estruturado, padronizando os sentimentos e as categorias.
2. **Visualização de Dados (Front-end):**
   * **Streamlit & Pandas:** O painel de controle interativo (`app.py`) lê os dados já processados pelo modelo de IA e os exibe por meio de gráficos de análise.
   * **UI Customizada:** Implementação de CSS customizado via Streamlit Markdown para injetar uma estética limpa, baseada em Glassmorphism e gradientes (estilo *Frutiger Aero*).

## Como Rodar o Projeto Localmente

**1. Instale as dependências:**
Bash
pip install -r requirements.txt

**2. Processe os dados brutais via IA:(Nota: Para rodar este script, é necessário inserir uma chave de API do OpenRouter no arquivo process_data.py. Por motivos de segurança, a chave não é enviada ao repositório).**
Bash
python process_data.py

**3. Inicie o Dashboard:**

Bash
python -m streamlit run app.py


## Estrutura do Projeto
```text
desafio-inovacao-fhazin
 ┣ .streamlit/
 ┃ ┗ config.toml           # Força o tema claro para a UI customizada
 ┣ app.py                  # Front-end: Dashboard Streamlit
 ┣ process_data.py         # Back-end: Pipeline de IA com Llama 3
 ┣ raw_reviews.json        # Dados brutos de entrada (Simulação)
 ┣ processed_reviews.json  # Base de dados estruturada (Gerada pela IA em process_data.py)
 ┣ requerimentos.txt       # Dependências do projeto
 ┗ README.md               # Documentação

