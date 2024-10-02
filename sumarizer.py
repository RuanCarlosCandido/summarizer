import os
from transformers import BartForConditionalGeneration, BartTokenizer
import torch

# Verificar se a GPU está disponível
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carregar o modelo e o tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name).to(device)

def partition_text(text, max_length=1024):
    # Dividir o texto em partes menores
    tokens = tokenizer.encode(text)
    for i in range(0, len(tokens), max_length):
        yield tokenizer.decode(tokens[i:i + max_length], skip_special_tokens=True)

def summarize(text):
    # Resumir o texto utilizando o modelo BART
    inputs = tokenizer(text, return_tensors='pt', max_length=1024, truncation=True)
    inputs = {key: val.to(device) for key, val in inputs.items()}
    summary_ids = model.generate(
        inputs['input_ids'], 
        max_length=3000,  # Aumentar o tamanho máximo
        min_length=100,  # Aumentar o tamanho mínimo
        length_penalty=2.0, 
        num_beams=4, 
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

def summarize_file(file_path):
    # Ler e resumir o arquivo de texto
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()

    summaries = []
    for part in partition_text(text):
        summaries.append(summarize(part))

    # Combinar os resumos de uma maneira coesa
    combined_summary = ' '.join(summaries)
    #combined_summary = summarize(combined_summary)  # Resumir novamente para coesão

    return combined_summary

# Caminho para o arquivo que você deseja resumir
file_path = 'transcription.txt'
summary = summarize_file(file_path)

# Salvar o resumo em um novo arquivo
with open('resumo.txt', 'w', encoding='utf-8') as file:
    file.write(summary)

print("Resumido com sucesso. Confira o arquivo 'resumo.txt'.")

