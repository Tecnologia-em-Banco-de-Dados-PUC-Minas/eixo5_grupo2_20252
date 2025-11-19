"""
Script de Análise de Resultados - Naive Bayes
Gera métricas, gráficos e relatórios comparativos dos modelos
"""

import sys
import io

# Configura encoding para Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import os
from datetime import datetime
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

from naive_bayes_pipeline import SentimentAnalyzer

# Configuração de estilo para gráficos
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except:
    try:
        plt.style.use('seaborn-darkgrid')
    except:
        plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# Cria pasta de resultados
RESULTADOS_DIR = 'analise_resultados'
os.makedirs(RESULTADOS_DIR, exist_ok=True)
os.makedirs(f'{RESULTADOS_DIR}/graficos', exist_ok=True)


class AnalisadorResultados:
    """Classe para análise e comparação de resultados dos modelos"""
    
    def __init__(self, df: pd.DataFrame):
        """
        Inicializa o analisador
        
        Args:
            df: DataFrame com os dados
        """
        self.df = df
        self.resultados = []
        self.modelos_treinados = {}
        
    def carregar_e_avaliar_modelo(self, caminho_modelo: str, test_size: float = 0.2, random_state: int = 42):
        """
        Carrega um modelo existente e avalia no conjunto de teste
        
        Args:
            caminho_modelo: Caminho para o arquivo do modelo (.pkl)
            test_size: Proporção de dados para teste
            random_state: Seed para reprodutibilidade
        """
        print("\n" + "="*70)
        print("CARREGANDO E AVALIANDO MODELO EXISTENTE")
        print("="*70)
        
        try:
            # Carrega o modelo
            print(f"\nCarregando modelo de: {caminho_modelo}")
            analyzer = SentimentAnalyzer()
            analyzer.load_model(caminho_modelo)
            
            # Prepara os dados (sem fazer fit novamente no vectorizer)
            print("\nPreparando dados para avaliação...")
            print("Iniciando pré-processamento dos dados...")
            
            # Limpa os textos
            df_temp = self.df.copy()
            df_temp['cleaned_text'] = analyzer.preprocessor.preprocess_corpus(df_temp['review'])
            df_temp = df_temp[df_temp['cleaned_text'].str.len() > 0]
            
            # Vetoriza usando apenas transform (não fit_transform)
            print(f"Vetorizando textos usando {analyzer.vectorization_method.upper()}...")
            X = analyzer.vectorizer.transform(df_temp['cleaned_text'])
            
            # Processa as labels usando o label_map do modelo
            y = df_temp['sentiment'].map(analyzer.label_map).values
            
            print(f"Shape dos dados: {X.shape}")
            print(f"Classes: {analyzer.label_map}")
            
            # Divide em treino e teste (mesma divisão usada no treinamento)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
            
            print(f"Teste: {X_test.shape[0]} amostras")
            
            # Avalia o modelo
            print("\nAvaliando modelo...")
            y_pred = analyzer.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            # Relatório de classificação
            reverse_label_map = {v: k for k, v in analyzer.label_map.items()}
            y_test_labels = [reverse_label_map[y] for y in y_test]
            y_pred_labels = [reverse_label_map[y] for y in y_pred]
            
            cm = confusion_matrix(y_test_labels, y_pred_labels, labels=list(reverse_label_map.values()))
            
            # Determina nome do modelo baseado nas configurações
            metodo = analyzer.vectorization_method.upper()
            features = analyzer.max_features
            nome = f"{metodo} ({features} features)"
            
            resultado = {
                'nome': nome,
                'vectorization_method': analyzer.vectorization_method,
                'max_features': analyzer.max_features,
                'accuracy': accuracy,
                'confusion_matrix': cm,
                'classification_report': classification_report(y_test_labels, y_pred_labels),
                'labels': list(reverse_label_map.values()),
                'modelo': analyzer
            }
            
            self.resultados.append(resultado)
            self.modelos_treinados[nome] = analyzer
            
            print(f"\n{'='*70}")
            print(f"Modelo: {nome}")
            print(f"Acurácia: {accuracy*100:.2f}%")
            print(f"{'='*70}\n")
            
        except FileNotFoundError:
            print(f"❌ ERRO: Arquivo '{caminho_modelo}' não encontrado!")
            raise
        except Exception as e:
            print(f"❌ ERRO ao carregar modelo: {e}")
            raise
    
    def treinar_modelos_comparacao(self, test_size: float = 0.2, random_state: int = 42):
        """
        Treina modelos com diferentes configurações para comparação
        
        Args:
            test_size: Proporção de dados para teste
            random_state: Seed para reprodutibilidade
        """
        print("\n" + "="*70)
        print("TREINANDO MODELOS PARA COMPARAÇÃO")
        print("="*70)
        
        # Configurações para testar
        configs = [
            {'vectorization_method': 'tfidf', 'max_features': 3000, 'nome': 'TF-IDF (3000 features)'},
            {'vectorization_method': 'tfidf', 'max_features': 5000, 'nome': 'TF-IDF (5000 features)'},
            {'vectorization_method': 'tfidf', 'max_features': 10000, 'nome': 'TF-IDF (10000 features)'},
            {'vectorization_method': 'count', 'max_features': 3000, 'nome': 'Count (3000 features)'},
            {'vectorization_method': 'count', 'max_features': 5000, 'nome': 'Count (5000 features)'},
            {'vectorization_method': 'count', 'max_features': 10000, 'nome': 'Count (10000 features)'},
        ]
        
        for config in configs:
            print(f"\n{'='*70}")
            print(f"Treinando: {config['nome']}")
            print(f"{'='*70}")
            
            try:
                analyzer = SentimentAnalyzer(
                    vectorization_method=config['vectorization_method'],
                    max_features=config['max_features']
                )
                
                metrics = analyzer.train(self.df, test_size=test_size, random_state=random_state)
                
                resultado = {
                    'nome': config['nome'],
                    'vectorization_method': config['vectorization_method'],
                    'max_features': config['max_features'],
                    'accuracy': metrics['accuracy'],
                    'confusion_matrix': metrics['confusion_matrix'],
                    'classification_report': metrics['classification_report'],
                    'labels': metrics['labels'],
                    'modelo': analyzer
                }
                
                self.resultados.append(resultado)
                self.modelos_treinados[config['nome']] = analyzer
                
                print(f"✓ {config['nome']} - Acurácia: {metrics['accuracy']*100:.2f}%")
                
            except Exception as e:
                print(f"✗ Erro ao treinar {config['nome']}: {e}")
        
        print(f"\n{'='*70}")
        print(f"Total de modelos treinados: {len(self.resultados)}")
        print(f"{'='*70}\n")
    
    def gerar_grafico_comparacao_acuracia(self):
        """Gera gráfico comparando acurácias dos modelos"""
        if not self.resultados:
            print("Nenhum resultado disponível para gerar gráfico.")
            return
        
        print("Gerando gráfico de comparação de acurácias...")
        
        nomes = [r['nome'] for r in self.resultados]
        acuracias = [r['accuracy'] * 100 for r in self.resultados]
        cores = ['#2ecc71' if 'TF-IDF' in n else '#e74c3c' for n in nomes]
        
        fig, ax = plt.subplots(figsize=(14, 8))
        bars = ax.barh(nomes, acuracias, color=cores, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        # Adiciona valores nas barras
        for i, (bar, acc) in enumerate(zip(bars, acuracias)):
            ax.text(acc + 0.2, i, f'{acc:.2f}%', 
                   va='center', fontsize=11, fontweight='bold')
        
        ax.set_xlabel('Acurácia (%)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Modelo', fontsize=12, fontweight='bold')
        ax.set_title('Comparação de Acurácia entre Modelos', fontsize=14, fontweight='bold', pad=20)
        ax.set_xlim([min(acuracias) - 2, max(acuracias) + 3])
        ax.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Legenda
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ecc71', label='TF-IDF'),
            Patch(facecolor='#e74c3c', label='Count Vectorizer')
        ]
        ax.legend(handles=legend_elements, loc='lower right', fontsize=11)
        
        plt.tight_layout()
        caminho = f'{RESULTADOS_DIR}/graficos/comparacao_acuracia.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {caminho}")
        plt.close()
    
    def gerar_matrizes_confusao(self):
        """Gera matrizes de confusão para todos os modelos"""
        if not self.resultados:
            print("Nenhum resultado disponível para gerar matrizes.")
            return
        
        print("Gerando matrizes de confusão...")
        
        n_modelos = len(self.resultados)
        
        # Para um único modelo, usa figura simples
        if n_modelos == 1:
            fig, ax = plt.subplots(figsize=(8, 6))
            axes = [ax]
        else:
            cols = 3
            rows = (n_modelos + cols - 1) // cols
            fig, axes = plt.subplots(rows, cols, figsize=(18, 6*rows))
            axes = axes.flatten()
        
        for idx, resultado in enumerate(self.resultados):
            cm = resultado['confusion_matrix']
            labels = resultado['labels']
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                       xticklabels=labels, yticklabels=labels,
                       ax=axes[idx], cbar_kws={'label': 'Quantidade'})
            
            axes[idx].set_title(f"{resultado['nome']}\nAcurácia: {resultado['accuracy']*100:.2f}%", 
                               fontsize=11, fontweight='bold')
            axes[idx].set_xlabel('Predito', fontsize=10)
            axes[idx].set_ylabel('Real', fontsize=10)
        
        # Remove eixos extras (apenas se houver múltiplos modelos)
        if n_modelos > 1:
            for idx in range(n_modelos, len(axes)):
                fig.delaxes(axes[idx])
        
        plt.tight_layout()
        caminho = f'{RESULTADOS_DIR}/graficos/matrizes_confusao.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        print(f"✓ Matrizes salvas: {caminho}")
        plt.close()
    
    def gerar_grafico_impacto_features(self):
        """Gera gráfico mostrando impacto do número de features"""
        if not self.resultados:
            print("Nenhum resultado disponível.")
            return
        
        print("Gerando gráfico de impacto de features...")
        
        # Agrupa por método de vetorização
        tfidf_data = {}
        count_data = {}
        
        for r in self.resultados:
            method = r['vectorization_method']
            max_feat = r['max_features']
            acc = r['accuracy'] * 100
            
            if method == 'tfidf':
                tfidf_data[max_feat] = acc
            else:
                count_data[max_feat] = acc
        
        fig, ax = plt.subplots(figsize=(12, 7))
        
        if tfidf_data:
            tfidf_feats = sorted(tfidf_data.keys())
            tfidf_accs = [tfidf_data[f] for f in tfidf_feats]
            ax.plot(tfidf_feats, tfidf_accs, marker='o', linewidth=2.5, 
                   markersize=10, label='TF-IDF', color='#2ecc71')
        
        if count_data:
            count_feats = sorted(count_data.keys())
            count_accs = [count_data[f] for f in count_feats]
            ax.plot(count_feats, count_accs, marker='s', linewidth=2.5, 
                   markersize=10, label='Count Vectorizer', color='#e74c3c')
        
        ax.set_xlabel('Número de Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
        ax.set_title('Impacto do Número de Features na Acurácia', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.legend(fontsize=11, loc='best')
        
        plt.tight_layout()
        caminho = f'{RESULTADOS_DIR}/graficos/impacto_features.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {caminho}")
        plt.close()
    
    def gerar_grafico_comparacao_metodos(self):
        """Gera gráfico comparando TF-IDF vs Count Vectorizer"""
        if not self.resultados:
            print("Nenhum resultado disponível.")
            return
        
        print("Gerando gráfico de comparação de métodos...")
        
        # Agrupa por número de features
        comparacao = {}
        
        for r in self.resultados:
            max_feat = r['max_features']
            method = r['vectorization_method']
            acc = r['accuracy'] * 100
            
            if max_feat not in comparacao:
                comparacao[max_feat] = {}
            comparacao[max_feat][method] = acc
        
        features = sorted(comparacao.keys())
        tfidf_accs = [comparacao[f].get('tfidf', 0) for f in features]
        count_accs = [comparacao[f].get('count', 0) for f in features]
        
        x = np.arange(len(features))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(12, 7))
        bars1 = ax.bar(x - width/2, tfidf_accs, width, label='TF-IDF', 
                      color='#2ecc71', alpha=0.8, edgecolor='black')
        bars2 = ax.bar(x + width/2, count_accs, width, label='Count Vectorizer', 
                      color='#e74c3c', alpha=0.8, edgecolor='black')
        
        # Adiciona valores nas barras
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                           f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
        
        ax.set_xlabel('Número de Features', fontsize=12, fontweight='bold')
        ax.set_ylabel('Acurácia (%)', fontsize=12, fontweight='bold')
        ax.set_title('Comparação TF-IDF vs Count Vectorizer', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{f}' for f in features])
        ax.legend(fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        plt.tight_layout()
        caminho = f'{RESULTADOS_DIR}/graficos/comparacao_metodos.png'
        plt.savefig(caminho, dpi=300, bbox_inches='tight')
        print(f"✓ Gráfico salvo: {caminho}")
        plt.close()
    
    def calcular_metricas_detalhadas(self) -> pd.DataFrame:
        """Calcula métricas detalhadas para cada modelo"""
        if not self.resultados:
            return pd.DataFrame()
        
        print("Calculando métricas detalhadas...")
        
        dados = []
        for r in self.resultados:
            cm = r['confusion_matrix']
            labels = r['labels']
            
            # Calcula métricas da matriz de confusão
            if len(labels) == 2:
                tn, fp, fn, tp = cm.ravel()
                
                precisao_pos = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall_pos = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1_pos = 2 * (precisao_pos * recall_pos) / (precisao_pos + recall_pos) if (precisao_pos + recall_pos) > 0 else 0
                
                precisao_neg = tn / (tn + fn) if (tn + fn) > 0 else 0
                recall_neg = tn / (tn + fp) if (tn + fp) > 0 else 0
                f1_neg = 2 * (precisao_neg * recall_neg) / (precisao_neg + recall_neg) if (precisao_neg + recall_neg) > 0 else 0
                
                erro = 1 - r['accuracy']
                
                dados.append({
                    'Modelo': r['nome'],
                    'Método': r['vectorization_method'],
                    'Features': r['max_features'],
                    'Acurácia (%)': f"{r['accuracy']*100:.2f}",
                    'Taxa de Erro (%)': f"{erro*100:.2f}",
                    'Precisão Positiva': f"{precisao_pos*100:.2f}",
                    'Recall Positivo': f"{recall_pos*100:.2f}",
                    'F1-Score Positivo': f"{f1_pos*100:.2f}",
                    'Precisão Negativa': f"{precisao_neg*100:.2f}",
                    'Recall Negativo': f"{recall_neg*100:.2f}",
                    'F1-Score Negativo': f"{f1_neg*100:.2f}",
                })
        
        df_metricas = pd.DataFrame(dados)
        caminho = f'{RESULTADOS_DIR}/metricas_detalhadas.csv'
        df_metricas.to_csv(caminho, index=False, encoding='utf-8-sig')
        print(f"✓ Métricas salvas: {caminho}")
        
        return df_metricas
    
    def gerar_relatorio_markdown(self):
        """Gera relatório markdown com análise completa"""
        if not self.resultados:
            print("Nenhum resultado disponível para relatório.")
            return
        
        print("Gerando relatório markdown...")
        
        # Encontra melhor modelo
        melhor_modelo = max(self.resultados, key=lambda x: x['accuracy'])
        
        # Calcula estatísticas
        acuracias = [r['accuracy'] for r in self.resultados]
        media_acuracia = np.mean(acuracias)
        std_acuracia = np.std(acuracias)
        
        # Compara métodos
        tfidf_results = [r for r in self.resultados if r['vectorization_method'] == 'tfidf']
        count_results = [r for r in self.resultados if r['vectorization_method'] == 'count']
        
        media_tfidf = np.mean([r['accuracy'] for r in tfidf_results]) if tfidf_results else 0
        media_count = np.mean([r['accuracy'] for r in count_results]) if count_results else 0
        
        relatorio = f"""# Relatório de Análise de Resultados - Naive Bayes

**Data de Geração:** {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}

---

## 📊 Resumo Executivo

Este relatório apresenta uma análise completa dos resultados obtidos pelos modelos de Naive Bayes para análise de sentimentos, comparando diferentes métodos de vetorização e configurações de parâmetros.

### Estatísticas Gerais

- **Total de Modelos Testados:** {len(self.resultados)}
- **Acurácia Média:** {media_acuracia*100:.2f}%
- **Desvio Padrão:** {std_acuracia*100:.2f}%
- **Melhor Modelo:** {melhor_modelo['nome']}
- **Melhor Acurácia:** {melhor_modelo['accuracy']*100:.2f}%

---

## 🏆 Melhor Modelo

**Configuração:**
- Método de Vetorização: {melhor_modelo['vectorization_method'].upper()}
- Número de Features: {melhor_modelo['max_features']}
- Acurácia: **{melhor_modelo['accuracy']*100:.2f}%**
- Taxa de Erro: **{(1 - melhor_modelo['accuracy'])*100:.2f}%**

---

## 📈 Comparação de Métodos de Vetorização

### TF-IDF vs Count Vectorizer

| Método | Acurácia Média | Modelos Testados |
|--------|----------------|------------------|
| **TF-IDF** | {media_tfidf*100:.2f}% | {len(tfidf_results)} |
| **Count Vectorizer** | {media_count*100:.2f}% | {len(count_results)} |

**Análise:** {'TF-IDF apresentou melhor performance média' if media_tfidf > media_count else 'Count Vectorizer apresentou melhor performance média' if media_count > media_tfidf else 'Ambos os métodos apresentaram performance similar'}.

---

## 🔍 Impacto do Número de Features

### Análise por Configuração

"""
        
        # Agrupa por número de features
        por_features = {}
        for r in self.resultados:
            n_feat = r['max_features']
            if n_feat not in por_features:
                por_features[n_feat] = []
            por_features[n_feat].append(r)
        
        for n_feat in sorted(por_features.keys()):
            modelos = por_features[n_feat]
            melhor = max(modelos, key=lambda x: x['accuracy'])
            relatorio += f"""
#### {n_feat} Features

- **Melhor Método:** {melhor['vectorization_method'].upper()}
- **Melhor Acurácia:** {melhor['accuracy']*100:.2f}%
- **Comparação:**
"""
            for m in modelos:
                relatorio += f"  - {m['vectorization_method'].upper()}: {m['accuracy']*100:.2f}%\n"
        
        relatorio += f"""
---

## 📋 Resultados Detalhados por Modelo

"""
        
        for i, resultado in enumerate(sorted(self.resultados, key=lambda x: x['accuracy'], reverse=True), 1):
            relatorio += f"""
### {i}. {resultado['nome']}

- **Acurácia:** {resultado['accuracy']*100:.2f}%
- **Taxa de Erro:** {(1 - resultado['accuracy'])*100:.2f}%
- **Método:** {resultado['vectorization_method'].upper()}
- **Features:** {resultado['max_features']}

#### Matriz de Confusão

```
{resultado['confusion_matrix']}
```

#### Relatório de Classificação

```
{resultado['classification_report']}
```

---
"""
        
        relatorio += f"""
## 💡 Interpretações e Conclusões

### 1. Método de Vetorização

"""
        
        if media_tfidf > media_count:
            relatorio += f"""
**TF-IDF** demonstrou ser superior ao Count Vectorizer para este problema, com uma diferença média de {abs(media_tfidf - media_count)*100:.2f} pontos percentuais. Isso sugere que a ponderação de termos por frequência inversa de documento ajuda a identificar melhor os padrões de sentimento, reduzindo o impacto de palavras muito comuns.
"""
        elif media_count > media_tfidf:
            relatorio += f"""
**Count Vectorizer** demonstrou ser superior ao TF-IDF para este problema, com uma diferença média de {abs(media_tfidf - media_count)*100:.2f} pontos percentuais. Isso pode indicar que a frequência bruta de palavras é mais relevante para análise de sentimentos neste dataset.
"""
        else:
            relatorio += """
Ambos os métodos apresentaram performance similar, sugerindo que a escolha pode depender de outros fatores como tempo de processamento ou interpretabilidade.
"""
        
        relatorio += f"""

### 2. Número de Features

"""
        
        # Analisa tendência de features
        if len(por_features) > 1:
            features_ordenadas = sorted(por_features.keys())
            melhor_por_feat = [max(por_features[f], key=lambda x: x['accuracy'])['accuracy'] 
                             for f in features_ordenadas]
            
            if melhor_por_feat[-1] > melhor_por_feat[0]:
                relatorio += f"""
Observa-se uma **tendência positiva** com o aumento do número de features. O modelo com {features_ordenadas[-1]} features apresentou melhor performance, sugerindo que mais features capturam melhor os padrões do texto. No entanto, é importante considerar o trade-off entre performance e tempo de processamento.
"""
            elif melhor_por_feat[-1] < melhor_por_feat[0]:
                relatorio += f"""
Observa-se que **menos features** podem ser suficientes para este problema. O modelo com {features_ordenadas[0]} features apresentou boa performance, sugerindo que um número intermediário de features pode ser ideal, evitando overfitting e reduzindo tempo de processamento.
"""
            else:
                relatorio += """
O número de features não apresentou impacto significativo na performance, sugerindo que o modelo é robusto a diferentes configurações deste parâmetro.
"""
        
        relatorio += f"""

### 3. Características do Melhor Modelo

O melhor modelo ({melhor_modelo['nome']}) alcançou uma acurácia de **{melhor_modelo['accuracy']*100:.2f}%**, demonstrando que:

1. A configuração de **{melhor_modelo['max_features']} features** é adequada para este problema
2. O método **{melhor_modelo['vectorization_method'].upper()}** é mais eficaz para este dataset
3. O algoritmo Naive Bayes é capaz de capturar padrões relevantes para análise de sentimentos

### 4. Recomendações

1. **Para Produção:** Utilizar o modelo {melhor_modelo['nome']} com {melhor_modelo['max_features']} features
2. **Para Experimentação:** Testar diferentes n-grams e técnicas de pré-processamento
3. **Para Otimização:** Considerar validação cruzada para seleção de hiperparâmetros
4. **Para Interpretabilidade:** Analisar as features mais importantes do modelo

---

## 📁 Arquivos Gerados

- `graficos/comparacao_acuracia.png` - Comparação de acurácias entre modelos
- `graficos/matrizes_confusao.png` - Matrizes de confusão de todos os modelos
- `graficos/impacto_features.png` - Impacto do número de features
- `graficos/comparacao_metodos.png` - Comparação TF-IDF vs Count Vectorizer
- `metricas_detalhadas.csv` - Tabela com todas as métricas calculadas

---

## 📝 Notas Finais

Esta análise fornece uma visão abrangente do desempenho dos modelos de Naive Bayes para análise de sentimentos. Os resultados demonstram a importância de testar diferentes configurações e métodos de vetorização para encontrar a melhor solução para o problema específico.

**Próximos Passos Sugeridos:**
- Testar outros algoritmos (SVM, Random Forest, etc.)
- Aplicar técnicas de balanceamento de classes se necessário
- Realizar análise de features mais importantes
- Implementar ensemble de modelos
"""
        
        caminho = f'{RESULTADOS_DIR}/RELATORIO_ANALISE.md'
        with open(caminho, 'w', encoding='utf-8') as f:
            f.write(relatorio)
        
        print(f"✓ Relatório salvo: {caminho}")
    
    def executar_analise_modelo_existente(self, caminho_modelo: str, test_size: float = 0.2, random_state: int = 42):
        """
        Executa análise completa de um modelo existente: carrega, avalia, gera gráficos e relatório
        
        Args:
            caminho_modelo: Caminho para o arquivo do modelo (.pkl)
            test_size: Proporção de dados para teste
            random_state: Seed para reprodutibilidade
        """
        print("\n" + "="*70)
        print("INICIANDO ANÁLISE DE MODELO EXISTENTE")
        print("="*70)
        
        # 1. Carrega e avalia modelo
        self.carregar_e_avaliar_modelo(caminho_modelo, test_size, random_state)
        
        if not self.resultados:
            print("Nenhum modelo foi carregado com sucesso. Abortando análise.")
            return
        
        # 2. Gera gráficos
        print("\n" + "="*70)
        print("GERANDO GRÁFICOS")
        print("="*70)
        self.gerar_grafico_comparacao_acuracia()
        self.gerar_matrizes_confusao()
        # Para um único modelo, alguns gráficos não fazem sentido
        # self.gerar_grafico_impacto_features()
        # self.gerar_grafico_comparacao_metodos()
        
        # 3. Calcula métricas
        print("\n" + "="*70)
        print("CALCULANDO MÉTRICAS")
        print("="*70)
        self.calcular_metricas_detalhadas()
        
        # 4. Gera relatório
        print("\n" + "="*70)
        print("GERANDO RELATÓRIO")
        print("="*70)
        self.gerar_relatorio_markdown()
        
        print("\n" + "="*70)
        print("✅ ANÁLISE COMPLETA FINALIZADA!")
        print("="*70)
        print(f"\nTodos os resultados foram salvos em: {RESULTADOS_DIR}/")
        print("\nArquivos gerados:")
        print("  - graficos/comparacao_acuracia.png")
        print("  - graficos/matrizes_confusao.png")
        print("  - metricas_detalhadas.csv")
        print("  - RELATORIO_ANALISE.md")
        print("="*70 + "\n")
    
    def executar_analise_completa(self, test_size: float = 0.2, random_state: int = 42):
        """
        Executa análise completa: treina modelos, gera gráficos e relatório
        
        Args:
            test_size: Proporção de dados para teste
            random_state: Seed para reprodutibilidade
        """
        print("\n" + "="*70)
        print("INICIANDO ANÁLISE COMPLETA DE RESULTADOS")
        print("="*70)
        
        # 1. Treina modelos
        self.treinar_modelos_comparacao(test_size, random_state)
        
        if not self.resultados:
            print("Nenhum modelo foi treinado com sucesso. Abortando análise.")
            return
        
        # 2. Gera gráficos
        print("\n" + "="*70)
        print("GERANDO GRÁFICOS")
        print("="*70)
        self.gerar_grafico_comparacao_acuracia()
        self.gerar_matrizes_confusao()
        self.gerar_grafico_impacto_features()
        self.gerar_grafico_comparacao_metodos()
        
        # 3. Calcula métricas
        print("\n" + "="*70)
        print("CALCULANDO MÉTRICAS")
        print("="*70)
        self.calcular_metricas_detalhadas()
        
        # 4. Gera relatório
        print("\n" + "="*70)
        print("GERANDO RELATÓRIO")
        print("="*70)
        self.gerar_relatorio_markdown()
        
        print("\n" + "="*70)
        print("✅ ANÁLISE COMPLETA FINALIZADA!")
        print("="*70)
        print(f"\nTodos os resultados foram salvos em: {RESULTADOS_DIR}/")
        print("\nArquivos gerados:")
        print("  - graficos/comparacao_acuracia.png")
        print("  - graficos/matrizes_confusao.png")
        print("  - graficos/impacto_features.png")
        print("  - graficos/comparacao_metodos.png")
        print("  - metricas_detalhadas.csv")
        print("  - RELATORIO_ANALISE.md")
        print("="*70 + "\n")


def main():
    """Função principal"""
    print("\n" + "="*70)
    print("ANÁLISE DE RESULTADOS - NAIVE BAYES")
    print("="*70)
    
    # Carrega dados
    print("\nCarregando dataset...")
    try:
        df = pd.read_csv('dados/dataset.csv')
        print(f"✓ Dataset carregado: {len(df)} amostras")
        print(f"Distribuição de sentimentos:\n{df['sentiment'].value_counts()}\n")
    except FileNotFoundError:
        print("❌ ERRO: Arquivo 'dados/dataset.csv' não encontrado!")
        print("Por favor, verifique se o arquivo existe no caminho correto.")
        return
    except Exception as e:
        print(f"❌ ERRO ao carregar dataset: {e}")
        return
    
    # Cria analisador e executa análise do modelo existente
    analisador = AnalisadorResultados(df)
    
    # Tenta carregar o modelo existente
    caminho_modelo = 'naive_bayes_model.pkl'
    if not os.path.exists(caminho_modelo):
        print(f"❌ ERRO: Modelo '{caminho_modelo}' não encontrado!")
        print("Por favor, treine um modelo primeiro executando: python naive_bayes_pipeline.py")
        return
    
    analisador.executar_analise_modelo_existente(caminho_modelo, test_size=0.2, random_state=42)


if __name__ == "__main__":
    main()

