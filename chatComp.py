from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Vou viajar para Fernando de Noronha em Setembro de 2024, durante 3 dias. Me faça um roteiro de viagens."}
  ]
)

print(response.choices[0].message.content)