import pandas as pd
import numpy as np
import re
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

df = pd.read_csv('spam.csv')

if 'CATEGORY' not in df.columns and 'category' in df.columns:
    df['CATEGORY'] = df['category']
if 'MESSAGE' not in df.columns and 'message' in df.columns:
    df['MESSAGE'] = df['message']

def load_stopwords(filepath='stopwords.txt'):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return set(f.read().splitlines())
    except FileNotFoundError:
        print(f"ERRO: Arquivo {filepath} não encontrado.")
        return set()
    except Exception as e:
        print(f"ERRO ao carregar stopwords: {e}")
        return set()

stop_words = load_stopwords()

def preprocess_text(text):
    if not isinstance(text, str):
        return ""
    

    text = text.lower()
    
    text = re.sub(r'[àáâãäå]', 'a', text)
    text = re.sub(r'[èéêë]', 'e', text)
    text = re.sub(r'[ìíîï]', 'i', text)
    text = re.sub(r'[òóôõö]', 'o', text)
    text = re.sub(r'[ùúûü]', 'u', text)
    text = re.sub(r'[ýÿ]', 'y', text)
    text = re.sub(r'[ç]', 'c', text)
    
    text = re.sub(r'<.*?>', ' ', text)
    
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    
    tokens = re.findall(r'\w+', text)
    
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    
    tokens = [token for token in tokens if token not in stop_words]
    
    tokens = [token for token in tokens if len(token) > 2]
    
    
    clean_text = ' '.join(tokens)
    
    return clean_text


df['preprocessed_text'] = df['MESSAGE'].apply(preprocess_text)


vectorizer = TfidfVectorizer(
    min_df=0.1,  
    max_df=0.9,  
    ngram_range=(1, 3)  
)

X = vectorizer.fit_transform(df['preprocessed_text'])
y = df['CATEGORY']


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

neural_network = MLPClassifier(
    hidden_layer_sizes=(100,),  
    activation='relu',          
    solver='adam',              
    alpha=0.0001,               
    max_iter=200,               
    random_state=42,
    verbose=True
)

neural_network.fit(X_train, y_train)

y_pred = neural_network.predict(X_test)
y_prob = neural_network.predict_proba(X_test)[:, 1]  

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')
auc = roc_auc_score(y_test, y_prob)
mcc = matthews_corrcoef(y_test, y_pred)

print("\nMétricas de desempenho:")
print(f"Acurácia (CA): {accuracy:.3f}")
print(f"Precisão (Prec): {precision:.3f}")
print(f"Recall: {recall:.3f}")
print(f"F1-Score (F1): {f1:.3f}")
print(f"Área sob a curva ROC (AUC): {auc:.3f}")
print(f"Coeficiente de Correlação de Matthews (MCC): {mcc:.3f}")

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.xlabel('Predição')
plt.ylabel('Real')
plt.title('Matriz de Confusão - Rede Neural')
plt.savefig('confusion_matrix_nn.png')
plt.close()


print("\nExemplos de classificações do modelo:")
amostra = pd.DataFrame({
    'Frase Original': df['MESSAGE'].iloc[y_test.index],
    'Frase Pré-processada': df['preprocessed_text'].iloc[y_test.index],
    'Categoria Real': y_test.values,
    'Categoria Prevista': y_pred
})


amostra = amostra.sample(100, random_state=42)

for i, row in amostra.iterrows():
    print("-" * 60)
    print(f" E-mail: {row['Frase Original'][:300]}...")
    print(f" Real: {row['Categoria Real']} |  Previsto: {row['Categoria Prevista']}")

