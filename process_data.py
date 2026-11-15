import json
from openai import OpenAI

#Inicializa o cliente OpenAI com a chave de API e a URL base
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key="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",
)

# carregar os dados brutos
with open("raw_reviews.json", "r", encoding="utf-8") as file:
    reviews = json.load(file)

processed_data = []

#faz um loop pelos dados brutos e envia cada item para o modelo de IA para análise
for item in reviews:
    response = client.chat.completions.create(
        model="openrouter/free", # Specifically requesting the free tier model
        messages=[
            {"role": "system", "content": "You are a strict data analyzer for the construction industry. Output ONLY raw JSON with exactly three keys. 1. 'sentiment' (Must be exactly: Positivo, Neutro, or Negativo). 2. 'category' (Must be exactly: Atraso de Obra, Acabamento, Atendimento, or Financeiro/Contrato). 3. 'summary' (A 3-word summary written IN PORTUGUESE). Do not use markdown blocks like ```json and NEVER output 'None'."},
            {"role": "user", "content": f"Read this review: {item['text']}"}
        ]
    )
    
    #limpa a resposta da IA para garantir que seja um JSON válido
    clean_text = response.choices[0].message.content.replace("```json", "").replace("```", "").strip()
    
    try:
        ai_output = json.loads(clean_text)
        ai_output["original_text"] = item["text"]
        processed_data.append(ai_output)
    except json.JSONDecodeError:
        print(f"Skipped a review because the AI didn't format it right: {item['text']}")

#registra os dados processados em um novo arquivo JSON
with open("processed_reviews.json", "w", encoding="utf-8") as file:
    json.dump(processed_data, file, ensure_ascii=False, indent=4)

print("Data processing complete! Check processed_reviews.json")