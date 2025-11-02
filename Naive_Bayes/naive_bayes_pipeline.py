"""
Pipeline de Análise de Sentimentos usando Naive Bayes
Desenvolvido para análise de sentimentos em reviews do IMDB
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import re
import pickle
from typing import Tuple, List, Dict
import warnings
warnings.filterwarnings('ignore')


class TextPreprocessor:
    """Classe para pré-processamento de texto"""
    
    def __init__(self):
        self.vectorizer = None
    
    def clean_text(self, text: str) -> str:
        """
        Limpa o texto removendo HTML tags, caracteres especiais e normalizando espaços
        
        Args:
            text: Texto a ser limpo
            
        Returns:
            Texto limpo
        """
        if not isinstance(text, str):
            return ""
        
        # Remove HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Remove caracteres especiais, mantém apenas letras e espaços
        text = re.sub(r'[^A-Za-z\s]', '', text)
        
        # Normaliza espaços múltiplos
        text = re.sub(r'\s+', ' ', text)
        
        # Converte para minúsculas
        text = text.lower().strip()
        
        return text
    
    def preprocess_corpus(self, texts: pd.Series) -> pd.Series:
        """
        Pré-processa uma série de textos
        
        Args:
            texts: Série de textos
            
        Returns:
            Série de textos pré-processados
        """
        return texts.apply(self.clean_text)


class SentimentAnalyzer:
    """
    Classe principal para análise de sentimentos usando Naive Bayes
    """
    
    def __init__(self, vectorization_method: str = 'tfidf', max_features: int = 5000):
        """
        Inicializa o analisador de sentimentos
        
        Args:
            vectorization_method: Método de vetorização ('tfidf' ou 'count')
            max_features: Número máximo de features para vetorização
        """
        self.preprocessor = TextPreprocessor()
        self.vectorization_method = vectorization_method
        self.max_features = max_features
        
        # Escolhe o vetorizador baseado no método
        if vectorization_method == 'tfidf':
            self.vectorizer = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2))
        else:
            self.vectorizer = CountVectorizer(max_features=max_features, ngram_range=(1, 2))
        
        self.model = MultinomialNB()
        self.label_map = None
        self.is_fitted = False
    
    def prepare_data(self, df: pd.DataFrame, text_column: str = 'review', 
                     label_column: str = 'sentiment') -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepara os dados para treinamento
        
        Args:
            df: DataFrame com os dados
            text_column: Nome da coluna com o texto
            label_column: Nome da coluna com as labels
            
        Returns:
            Tupla (X, y) com features e labels
        """
        print("Iniciando pré-processamento dos dados...")
        
        # Limpa os textos
        df['cleaned_text'] = self.preprocessor.preprocess_corpus(df[text_column])
        
        # Remove linhas com texto vazio
        df = df[df['cleaned_text'].str.len() > 0]
        
        # Vetoriza os textos
        print(f"Vetorizando textos usando {self.vectorization_method.upper()}...")
        X = self.vectorizer.fit_transform(df['cleaned_text'])
        
        # Processa as labels
        if self.label_map is None:
            unique_labels = df[label_column].unique()
            self.label_map = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        
        y = df[label_column].map(self.label_map).values
        
        print(f"Shape dos dados: {X.shape}")
        print(f"Classes: {self.label_map}")
        
        return X, y
    
    def train(self, df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Dict:
        """
        Treina o modelo de Naive Bayes
        
        Args:
            df: DataFrame com os dados de treinamento
            test_size: Proporção de dados para teste
            random_state: Seed para reprodutibilidade
            
        Returns:
            Dicionário com métricas de avaliação
        """
        print("\n" + "="*60)
        print("TREINAMENTO DO MODELO NAIVE BAYES")
        print("="*60)
        
        # Prepara os dados
        X, y = self.prepare_data(df)
        
        # Divide em treino e teste
        print(f"\nDividindo dados em treino/teste (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        print(f"Treino: {X_train.shape[0]} amostras")
        print(f"Teste: {X_test.shape[0]} amostras")
        
        # Treina o modelo
        print("\nTreinando modelo Naive Bayes...")
        self.model.fit(X_train, y_train)
        self.is_fitted = True
        
        # Avalia o modelo
        print("\nAvaliando modelo...")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"ACURÁCIA: {accuracy*100:.2f}%")
        print(f"{'='*60}\n")
        
        # Relatório de classificação
        reverse_label_map = {v: k for k, v in self.label_map.items()}
        y_test_labels = [reverse_label_map[y] for y in y_test]
        y_pred_labels = [reverse_label_map[y] for y in y_pred]
        
        print("RELATÓRIO DE CLASSIFICAÇÃO:")
        print("-"*60)
        print(classification_report(y_test_labels, y_pred_labels))
        
        # Matriz de confusão
        print("\nMATRIZ DE CONFUSÃO:")
        print("-"*60)
        cm = confusion_matrix(y_test_labels, y_pred_labels, labels=list(reverse_label_map.values()))
        print(cm)
        
        # Retorna métricas
        metrics = {
            'accuracy': accuracy,
            'classification_report': classification_report(y_test_labels, y_pred_labels),
            'confusion_matrix': cm,
            'labels': list(reverse_label_map.values())
        }
        
        return metrics
    
    def predict(self, texts: List[str]) -> List[str]:
        """
        Faz predições em uma lista de textos
        
        Args:
            texts: Lista de textos para predição
            
        Returns:
            Lista de sentimentos previstos
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")
        
        # Pré-processa
        df = pd.DataFrame({'text': texts})
        df['cleaned_text'] = self.preprocessor.preprocess_corpus(df['text'])
        
        # Vetoriza
        X = self.vectorizer.transform(df['cleaned_text'])
        
        # Prediz
        y_pred = self.model.predict(X)
        
        # Converte de volta para labels
        reverse_label_map = {v: k for k, v in self.label_map.items()}
        predictions = [reverse_label_map[y] for y in y_pred]
        
        return predictions
    
    def predict_proba(self, texts: List[str]) -> np.ndarray:
        """
        Retorna as probabilidades de cada classe
        
        Args:
            texts: Lista de textos
            
        Returns:
            Array com probabilidades
        """
        if not self.is_fitted:
            raise ValueError("Modelo não foi treinado. Execute train() primeiro.")
        
        # Pré-processa
        df = pd.DataFrame({'text': texts})
        df['cleaned_text'] = self.preprocessor.preprocess_corpus(df['text'])
        
        # Vetoriza
        X = self.vectorizer.transform(df['cleaned_text'])
        
        # Retorna probabilidades
        return self.model.predict_proba(X)
    
    def save_model(self, filepath: str):
        """
        Salva o modelo em disco
        
        Args:
            filepath: Caminho para salvar o modelo
        """
        if not self.is_fitted:
            raise ValueError("Nenhum modelo treinado para salvar.")
        
        model_data = {
            'model': self.model,
            'vectorizer': self.vectorizer,
            'label_map': self.label_map,
            'vectorization_method': self.vectorization_method,
            'max_features': self.max_features
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Modelo salvo em: {filepath}")
    
    def load_model(self, filepath: str):
        """
        Carrega um modelo salvo
        
        Args:
            filepath: Caminho do modelo salvo
        """
        with open(filepath, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.vectorizer = model_data['vectorizer']
        self.label_map = model_data['label_map']
        self.vectorization_method = model_data.get('vectorization_method', 'tfidf')
        self.max_features = model_data.get('max_features', 5000)
        self.is_fitted = True
        
        print(f"Modelo carregado de: {filepath}")


def main():
    """
    Função principal para executar o pipeline
    """
    print("\n" + "="*60)
    print("PIPELINE DE ANÁLISE DE SENTIMENTOS - NAIVE BAYES")
    print("="*60 + "\n")
    
    # Carrega os dados
    print("Carregando dataset...")
    df = pd.read_csv('dados/dataset.csv')
    print(f"Dataset carregado: {len(df)} amostras")
    print(f"Distribuição de sentimentos:\n{df['sentiment'].value_counts()}\n")
    
    # Cria o analisador
    # Teste com TF-IDF primeiro (geralmente melhor performance)
    analyzer = SentimentAnalyzer(vectorization_method='tfidf', max_features=5000)
    
    # Treina o modelo
    metrics = analyzer.train(df, test_size=0.2, random_state=42)
    
    # Salva o modelo
    analyzer.save_model('naive_bayes_model.pkl')
    
    # Testa com alguns exemplos
    print("\n" + "="*60)
    print("TESTE COM EXEMPLOS")
    print("="*60)
    
    test_texts = [
        "This movie is absolutely fantastic! I loved every minute of it.",
        "Terrible movie, waste of time. Don't watch it.",
        "The acting was okay but the plot was confusing.",
        "Amazing cinematography and great soundtrack. Highly recommended!"
    ]
    
    predictions = analyzer.predict(test_texts)
    probabilities = analyzer.predict_proba(test_texts)
    
    for i, text in enumerate(test_texts):
        print(f"\nTexto: {text[:80]}...")
        print(f"Sentimento: {predictions[i]}")
        print(f"Probabilidades: {dict(zip(analyzer.label_map.keys(), probabilities[i]))}")
    
    print("\n" + "="*60)
    print("PIPELINE CONCLUÍDO COM SUCESSO!")
    print("="*60)


if __name__ == "__main__":
    main()

