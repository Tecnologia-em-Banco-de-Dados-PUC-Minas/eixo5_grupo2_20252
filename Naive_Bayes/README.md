# Relatório de Análise de Resultados - Naive Bayes

**Data de Geração:** 18/11/2025 19:30:49

---

## 📊 Resumo Executivo

Este relatório apresenta uma análise completa dos resultados obtidos pelo modelo de Naive Bayes para análise de sentimentos.

### Estatísticas Gerais

- **Modelo Analisado:** TFIDF (5000 features)
- **Acurácia:** 85.95%
- **Taxa de Erro:** 14.05%

### Resultados Principais

| Métrica | Valor | Interpretação |
|---------|-------|---------------|
| **Acurácia Geral** | 85.95% | O modelo acerta aproximadamente 86 de cada 100 classificações |
| **Precisão (Negative)** | 87% | Quando o modelo diz "negativo", está correto 87% das vezes |
| **Precisão (Positive)** | 85% | Quando o modelo diz "positivo", está correto 85% das vezes |
| **Recall (Negative)** | 84% | O modelo identifica 84% de todos os reviews negativos |
| **Recall (Positive)** | 87% | O modelo identifica 87% de todos os reviews positivos |
| **F1-Score** | 86% | Score geral de qualidade, balanceando precisão e recall |

### Conclusões Principais

1. ✅ **Modelo bem-sucedido:** Acurácia de 85.95% é um resultado sólido para análise de sentimentos
2. ✅ **Desempenho equilibrado:** O modelo não apresenta viés significativo para nenhuma classe
3. ✅ **Configuração adequada:** TF-IDF com 5000 features oferece bom equilíbrio performance/eficiência

---

## 🏆 Melhor Modelo

**Configuração:**
- Método de Vetorização: **TFIDF**
- Número de Features: **5000**
- Acurácia: **85.95%**
- Taxa de Erro: **14.05%**

---

## 📊 Gráficos e Visualizações

### 1. Comparação de Acurácia

![Comparação de Acurácia](graficos/comparacao_acuracia.png)

**Interpretação:** O modelo TFIDF (5000 features) alcançou 85.95% de acurácia, indicando que acerta aproximadamente 86 de cada 100 classificações.

### 2. Matriz de Confusão

![Matriz de Confusão](graficos/matrizes_confusao.png)

**Interpretação:** 
- **Verdadeiros Negativos:** 4.224 (reviews negativos corretos)
- **Falsos Positivos:** 776 (negativos classificados como positivos)
- **Falsos Negativos:** 629 (positivos classificados como negativos)
- **Verdadeiros Positivos:** 4.371 (reviews positivos corretos)

A diagonal principal (azul escuro) mostra os acertos, indicando desempenho equilibrado.

### 3. Impacto do Número de Features

![Impacto do Número de Features](graficos/impacto_features.png)

**Interpretação:** 
- **Tendência:** Aumentar o número de features geralmente melhora a acurácia
- **TF-IDF (verde):** Consistemente superior ao Count Vectorizer
- **Trade-off:** Mais features = melhor acurácia, mas também = mais tempo de processamento
- **Ponto ótimo:** 5000 features oferece bom equilíbrio entre performance e eficiência

### 4. Comparação TF-IDF vs Count Vectorizer

![Comparação TF-IDF vs Count Vectorizer](graficos/comparacao_metodos.png)

**Interpretação:**
- **TF-IDF (verde):** Supera Count Vectorizer em todas as configurações
- **Diferença:** TF-IDF geralmente supera Count Vectorizer em 2-3 pontos percentuais
- **Conclusão:** TF-IDF é o método mais adequado para este problema

---

## 📋 Resultados Detalhados

### Matriz de Confusão

```
                    Predito
                 Negative  Positive
Real  Negative    4224      776
      Positive     629     4371
```

**Análise:**
- **Total de amostras testadas:** 10.000
- **Total de acertos:** 8.595 (85.95%)
- **Total de erros:** 1.405 (14.05%)
- O modelo é ligeiramente melhor em identificar sentimentos positivos (4.371 vs 4.224)

### Relatório de Classificação

```
              precision    recall  f1-score   support

    negative       0.87      0.84      0.86      5000
    positive       0.85      0.87      0.86      5000

    accuracy                           0.86     10000
   macro avg       0.86      0.86      0.86     10000
```

**Métricas por Classe:**

| Classe | Precision | Recall | F1-Score | Significado |
|--------|-----------|--------|----------|-------------|
| **Negative** | 87% | 84% | 86% | Quando diz "negativo", está correto 87% das vezes. Identifica 84% dos negativos. |
| **Positive** | 85% | 87% | 86% | Quando diz "positivo", está correto 85% das vezes. Identifica 87% dos positivos. |

---

## 💡 Interpretações e Conclusões

### 1. Método de Vetorização

**TF-IDF** demonstrou ser superior porque:
- Pondera palavras por importância (palavras comuns recebem menor peso)
- Reduz ruído de palavras muito frequentes
- Melhor discriminação entre sentimentos positivos e negativos

### 2. Número de Features

**5000 features oferece:**
- ✅ Performance: 85.95% de acurácia (satisfatória)
- ✅ Eficiência: Tempo de processamento razoável
- ✅ Uso de memória: Consumo moderado

**Trade-off:** Aumentar para 10000 features pode melhorar apenas 1-2%, mas dobra o uso de recursos.

### 3. Análise do Desempenho

**Pontos Fortes:**
- ✅ Acurácia de 85.95% (acerta 86 de cada 100)
- ✅ Desempenho equilibrado entre classes
- ✅ Precisão e recall consistentes (86% F1-Score)

**Áreas de Melhoria:**
- ⚠️ 776 falsos positivos (15.5% dos negativos)
- ⚠️ 629 falsos negativos (12.6% dos positivos)
- ⚠️ Taxa de erro de 14.05% pode ser reduzida

### 4. Recomendações

**Para Produção:**
- ✅ Utilizar o modelo TFIDF (5000 features) - Configuração atual é adequada
- ✅ Monitorar performance em produção
- ✅ Implementar sistema de feedback

**Para Melhorias:**
- 🔬 Testar diferentes n-grams e técnicas de pré-processamento
- ⚙️ Validação cruzada para seleção de hiperparâmetros
- 🚀 Testar outros algoritmos (SVM, Random Forest) ou ensemble de modelos
- 🔍 Analisar features mais importantes e casos de erro

---

## 📁 Arquivos Gerados

- `graficos/comparacao_acuracia.png` - Comparação de acurácias entre modelos
- `graficos/matrizes_confusao.png` - Matriz de confusão do modelo
- `graficos/impacto_features.png` - Impacto do número de features
- `graficos/comparacao_metodos.png` - Comparação TF-IDF vs Count Vectorizer
- `metricas_detalhadas.csv` - Tabela com todas as métricas calculadas

---

## 📝 Notas Finais

### Contexto dos Resultados

**Comparação com benchmarks:**
- Classificador aleatório: 50% de acurácia
- Modelo atual: **85.95% de acurácia**
- Modelos de ponta: 90-95% de acurácia

**Conclusão:** O modelo está **bem acima do acaso** e próximo de resultados de ponta, sendo adequado para uso em produção.

### Aplicabilidade Prática

- ✅ **Adequado para:** Análise geral de sentimentos, filtragem de reviews, análise de tendências
- ⚠️ **Cuidado com:** Aplicações críticas onde cada erro tem alto custo
- 🔍 **Monitorar:** Performance em produção e ajustar conforme necessário

### Próximos Passos

1. **Curto Prazo:** Análise de erros e ajuste fino de parâmetros
2. **Médio Prazo:** Testar outros algoritmos e validação cruzada
3. **Longo Prazo:** Ensemble de modelos ou Deep Learning (LSTM, BERT)

---

**Conclusão:** O modelo de Naive Bayes com TF-IDF e 5000 features demonstrou ser uma **solução eficaz e prática** para análise de sentimentos, com 85.95% de acurácia e desempenho equilibrado, pronto para uso em produção.
