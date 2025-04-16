# Classificador de Spam usando Redes Neurais

Este projeto implementa um classificador de e-mails para detectar mensagens de spam utilizando técnicas de processamento de linguagem natural (NLP) e redes neurais artificiais.

## Descrição

O sistema analisa mensagens de e-mail e as classifica como spam ou não-spam (ham) utilizando um modelo de rede neural MLP (Multi-Layer Perceptron). O processo inclui:

1. Pré-processamento de texto
2. Vetorização usando TF-IDF
3. Treinamento de uma rede neural
4. Avaliação do modelo com diversas métricas

## Requisitos

- Python 3.6+
- Bibliotecas Python listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```
git clone https://github.com/seu-usuario/spam-classifier.git
cd spam-classifier
```

2. Instale as dependências:
```
pip install -r requirements.txt
```

3. Baixe os recursos do NLTK necessários:
```python
import nltk
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
```

## Dataset

O projeto utiliza um arquivo CSV chamado `spam.csv` que deve conter, no mínimo, as seguintes colunas:
- `MESSAGE` ou `message`: O conteúdo do e-mail
- `CATEGORY` ou `category`: A classificação (spam ou não-spam)

## Arquivo de Stopwords

O sistema tenta carregar um arquivo `stopwords.txt` contendo palavras a serem ignoradas durante o processamento. Este arquivo deve conter uma palavra por linha.

## Uso

Execute o script principal:
```
python spam_classifier.py
```

O script irá:
1. Carregar e pré-processar os dados
2. Treinar o modelo
3. Avaliar o desempenho
4. Gerar uma matriz de confusão em `confusion_matrix_nn.png`
5. Mostrar exemplos de classificações realizadas pelo modelo

## Funcionalidades

### Pré-processamento de Texto
- Normalização de caracteres com acentos
- Remoção de tags HTML
- Remoção de URLs
- Lematização
- Remoção de stopwords
- Remoção de tokens muito curtos

### Vetorização
- TF-IDF com n-gramas (1 a 3)
- Filtragem de termos muito raros ou muito comuns

### Modelo
- Rede Neural MLP com camada oculta de 100 neurônios
- Função de ativação ReLU
- Otimizador Adam

### Métricas de Avaliação
- Acurácia (CA)
- Precisão (Prec)
- Recall
- F1-Score (F1)
- Área sob a curva ROC (AUC)
- Coeficiente de Correlação de Matthews (MCC)
- Matriz de Confusão

## Estrutura do Projeto

```
spam-classifier/
├── spam_classifier.py     # Script principal
├── spam.csv               # Dataset (não incluído no repositório)
├── stopwords.txt          # Lista de stopwords
├── requirements.txt       # Dependências do projeto
├── README.md              # Este arquivo
└── confusion_matrix_nn.png # Matriz de confusão gerada (após execução)
```

## Resultados

O modelo gera uma matriz de confusão e imprime as métricas de desempenho no console. Além disso, exibe 100 exemplos aleatórios de classificações realizadas, permitindo uma análise qualitativa do desempenho.

## Contribuição

Sinta-se à vontade para contribuir com melhorias no modelo, no processo de pré-processamento ou na documentação. Abra um issue ou envie um pull request.

## Licença

[Especifique sua licença aqui, e.g., MIT, GPL, etc.]