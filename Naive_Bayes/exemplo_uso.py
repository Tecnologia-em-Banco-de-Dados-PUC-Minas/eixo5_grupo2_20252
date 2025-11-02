"""
Exemplo de uso do pipeline de análise de sentimentos
"""

from naive_bayes_pipeline import SentimentAnalyzer
import pandas as pd

def exemplo_predicao_individual():
    """Exemplo de predição em textos individuais"""
    print("\n" + "="*60)
    print("EXEMPLO 1: Predição Individual")
    print("="*60)
    
    # Carrega modelo treinado
    analyzer = SentimentAnalyzer()
    analyzer.load_model('naive_bayes_model.pkl')
    
    # Textos para teste
    textos = [
        "The movie was absolutely brilliant! Best film I've seen this year.",
        "Boring and predictable. I fell asleep halfway through.",
        "It's okay, nothing special but entertaining enough.",
        "Outstanding performance by the cast and a gripping storyline."
    ]
    
    # Faz predições
    for texto in textos:
        predicao = analyzer.predict([texto])[0]
        probabilidades = analyzer.predict_proba([texto])[0]
        
        # Converte numpy arrays para dicionário legível
        prob_dict = dict(zip(analyzer.label_map.keys(), probabilidades))
        
        print(f"\nTexto: {texto}")
        print(f"Sentimento: {predicao}")
        print(f"Confiança:")
        for sentiment, prob in prob_dict.items():
            print(f"  {sentiment}: {prob*100:.2f}%")


def exemplo_comparacao_modelos():
    """Exemplo comparando TF-IDF vs Count Vectorizer"""
    print("\n" + "="*60)
    print("EXEMPLO 2: Comparação TF-IDF vs Count Vectorizer")
    print("="*60)
    
    # Carrega amostra dos dados para teste rápido
    df = pd.read_csv('dados/dataset.csv').sample(10000, random_state=42)
    
    # Modelo com TF-IDF
    print("\n[1] Treinando com TF-IDF...")
    analyzer_tfidf = SentimentAnalyzer(vectorization_method='tfidf', max_features=3000)
    metrics_tfidf = analyzer_tfidf.train(df, test_size=0.3, random_state=42)
    
    # Modelo com Count Vectorizer
    print("\n[2] Treinando com Count Vectorizer...")
    analyzer_count = SentimentAnalyzer(vectorization_method='count', max_features=3000)
    metrics_count = analyzer_count.train(df, test_size=0.3, random_state=42)
    
    # Compara
    print("\n" + "="*60)
    print("COMPARAÇÃO:")
    print("="*60)
    print(f"TF-IDF Accuracy: {metrics_tfidf['accuracy']*100:.2f}%")
    print(f"Count Accuracy:  {metrics_count['accuracy']*100:.2f}%")
    print("="*60)


def exemplo_analise_batch():
    """Exemplo de análise em lote"""
    print("\n" + "="*60)
    print("EXEMPLO 3: Análise em Lote")
    print("="*60)
    
    # Carrega modelo
    analyzer = SentimentAnalyzer()
    analyzer.load_model('naive_bayes_model.pkl')
    
    # Simula reviews de um usuário
    reviews_usuario = [
        "Great movie, loved it!",
        "Not bad, but could be better.",
        "Terrible, worst movie ever!",
        "Amazing cinematography!",
        "Boring and long."
    ]
    
    # Faz predições em lote
    print("\nAnalisando reviews...")
    prediccoes = analyzer.predict(reviews_usuario)
    probabilidades = analyzer.predict_proba(reviews_usuario)
    
    # Estatísticas
    positivos = prediccoes.count('positive')
    negativos = prediccoes.count('negative')
    
    print(f"\nResultados:")
    print(f"  Positivos: {positivos} ({positivos/len(reviews_usuario)*100:.0f}%)")
    print(f"  Negativos: {negativos} ({negativos/len(reviews_usuario)*100:.0f}%)")
    
    # Mostra detalhes
    print("\nDetalhes por review:")
    for i, (review, pred, prob) in enumerate(zip(reviews_usuario, prediccoes, probabilidades), 1):
        confianca = max(prob) * 100
        print(f"\n{i}. {review}")
        print(f"   Sentimento: {pred} (confiança: {confianca:.1f}%)")


def exemplo_carregar_e_usar():
    """Exemplo simples de carregamento e uso"""
    print("\n" + "="*60)
    print("EXEMPLO 4: Uso Simplificado")
    print("="*60)
    
    # Carrega modelo existente
    analyzer = SentimentAnalyzer()
    analyzer.load_model('naive_bayes_model.pkl')
    
    # Testa um texto
    meu_texto = "This is the best movie I have ever watched!"
    sentimento = analyzer.predict([meu_texto])[0]
    
    print(f"\nTexto: '{meu_texto}'")
    print(f"Sentimento detectado: {sentimento}")
    

if __name__ == "__main__":
    print("\n🎬 EXEMPLOS DE USO DO PIPELINE DE ANÁLISE DE SENTIMENTOS")
    print("=" * 60)
    
    try:
        # Exemplo 1: Predição individual
        exemplo_predicao_individual()
        
        # Exemplo 2: Comparação de modelos (comentado para rapidez)
        # exemplo_comparacao_modelos()
        
        # Exemplo 3: Análise em lote
        exemplo_analise_batch()
        
        # Exemplo 4: Uso simplificado
        exemplo_carregar_e_usar()
        
        print("\n" + "="*60)
        print("✅ TODOS OS EXEMPLOS EXECUTADOS COM SUCESSO!")
        print("="*60)
        
    except FileNotFoundError:
        print("\n❌ ERRO: Modelo não encontrado!")
        print("Execute primeiro: python naive_bayes_pipeline.py")
    except Exception as e:
        print(f"\n❌ ERRO: {e}")

