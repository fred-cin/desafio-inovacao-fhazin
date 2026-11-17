# DCL — Inteligência de reputação para construtoras

Submissão para o Desafio de Inovação do PS do CITi 2026.2. Documentação completa em [`docs/DCL_Documentacao.pdf`](docs/DCL_Documentacao.pdf).

🔗 **[Abrir o painel](https://desafio-inovacao-fhazin.streamlit.app/)** — funciona sem login e sem chave (modo demonstração).

## O problema

Uma construtora vende um produto de centenas de milhares de reais com anos de antecedência. Nesse intervalo, o cliente
fala dela no Google, no Reclame Aqui e no Instagram, e o próximo comprador lê tudo antes de visitar o stand.
Mesmo assim, a maioria das construtoras regionais acompanha essas avaliações à mão: alguém lê, copia para uma planilha
e, quando dá, responde. Ninguém sabe dizer se o problema de prazo piorou neste trimestre, se a assistência técnica
está pior que a do concorrente ou qual reclamação vai virar processo.

## A solução

O DCL transforma avaliações soltas em decisões:

| Pergunta da diretoria | Onde o DCL responde |
|---|---|
| Nossa reputação está melhorando ou piorando? | **Visão geral** — Índice de Reputação (lógica de NPS) com variação de 90 dias e tendência por trimestre vs. média do mercado |
| Do que exatamente os clientes reclamam e o que elogiam? | **Visão geral** — análise *por aspecto*: uma avaliação pode elogiar o acabamento e reclamar do prazo |
| Como estamos contra a concorrência? | **Comparativo** — ranking, mapa de calor por categoria e "maiores distâncias para o melhor concorrente" |
| O que responder primeiro? | **Alertas e respostas** — fila por urgência, sinal de risco jurídico (Procon, advogado, distrato) e rascunho de resposta |
| Dá para usar com os meus dados? | **Analisar avaliações** — cole textos ou importe um CSV; a IA classifica na hora e o resultado entra no painel |

Construtoras são o primeiro mercado, não um exemplo: ticket alto, ciclo de venda longo e reputação que pesa direto
na venda na planta. A mesma arquitetura atende outros setores depois, mudando só a taxonomia de categorias.

## Como a IA é usada no produto

```
CSV / texto colado ──► classificação por aspecto (LLM via OpenRouter) ──► validação ──► métricas ──► painel
                         │  temperatura 0, saída JSON                      │ rejeita formato inválido
                         │  taxonomia única em dcl/taxonomy.py             │ e devolve o erro ao modelo
                         └─ sem chave: regras locais (modo demonstração)   └ até 3 tentativas
```

Para cada avaliação o modelo devolve: sentimento geral, aspectos citados (cada um com o próprio sentimento),
urgência, risco jurídico com justificativa e um resumo factual. A IA também escreve o **resumo executivo** (só com
números calculados pelo sistema, sem inventar dados) e os **rascunhos de resposta** — sempre revisados por uma pessoa.

### Qualidade medida, não suposta

`evaluate.py` compara a classificação com rótulos de referência feitos à mão (`data/rotulos_referencia.csv`) e salva
o resultado em `docs/avaliacao.json`, exibido na aba *Como funciona*.

| Motor | Sentimento (acurácia) | Categorias (F1) | Aspecto completo (F1) | Risco jurídico (recall) |
|---|---|---|---|---|
| Regras locais (linha de base) | 89% | 89% | 82% | 100% |
| LLM (Llama 3.3 70B) | 100% | 95% | 93% | 100% |

O modelo supera as regras em tudo, mas a precisão do risco jurídico é de só 35% (recall 100%): ele sinaliza mais casos que os rótulos, o que infla a fila de urgência. Os 100% de sentimento vêm de 66 avaliações rotuladas pelo mesmo autor dos dados, ou seja, um conjunto fácil. As regras locais foram ajustadas neste mesmo conjunto, então o número delas é otimista. Elas existem para o painel
nunca quebrar sem chave e para servir de linha de base: se o modelo não supera as regras, não vale o custo.

## Rodando localmente

```bash
pip install -r requirements.txt
streamlit run app.py                       # abre em modo demonstração
```

Para ligar a IA, copie `.streamlit/secrets.toml.example` para `.streamlit/secrets.toml` e preencha a chave do
[OpenRouter](https://openrouter.ai/keys) (o arquivo está no `.gitignore`). Nos scripts de linha de comando,
use a variável de ambiente:

```bash
export OPENROUTER_API_KEY=sk-or-...
python process_data.py                     # reclassifica data/raw_reviews.csv com a IA
python evaluate.py                         # mede a qualidade contra os rótulos de referência
```

No Streamlit Cloud, a chave vai em *App settings → Secrets*, no mesmo formato do exemplo.

## Dados

As construtoras do conjunto de demonstração (Maré Alta, Baobá e Vértice) são **fictícias** e as avaliações foram
**simuladas** para o protótipo — o painel avisa isso no topo. Para dados reais, use o modelo
`data/modelo_importacao.csv` (colunas aceitas: texto/avaliação/comentário, empresa, fonte, data).

## Estrutura

```text
app.py                  Painel Streamlit (5 abas)
process_data.py         Pipeline: CSV bruto -> base classificada (JSON)
evaluate.py             Mede acurácia contra rótulos de referência
dcl/
  taxonomy.py           Categorias, sentimentos e validação da saída da IA
  llm.py                Cliente OpenRouter: chave por ambiente/secrets, JSON, re-tentativas
  analyzer.py           Prompt de classificação por aspecto + fallback para regras
  heuristic.py          Regras locais (modo demonstração e linha de base)
  insights.py           Índice de Reputação, comparativos, alertas, resumo e respostas
  data.py               Importação de CSV tolerante, base processada, formatos derivados
  charts.py             Gráfico de menções por categoria (testável fora do app)
data/                   Avaliações, base processada, rótulos de referência, modelo de importação
docs/DCL_Documentacao.pdf  Documentação completa (plataforma, funcionalidades, tecnologias, uso de IA)
docs/avaliacao.json     Resultado da última avaliação de qualidade
```

## Tecnologias

Python, Streamlit, pandas, Altair, OpenRouter (API compatível com OpenAI; modelo padrão
`meta-llama/llama-3.3-70b-instruct`, configurável por `DCL_MODEL`).
