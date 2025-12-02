# Eixo 5 | Arquitetura de Dados
## Pipeline de Sentimentos (IMDB)

### Visão Geral
Pipeline de análise de sentimentos desenvolvido na graduação em **Tecnologia em Banco de Dados (2025/2)**.  
Nesta nova versão, o projeto foi **refatorado para execução direta em Google Colab**, eliminando a dependência de Databricks e AWS.  
O pipeline coleta, limpa e processa avaliações públicas de filmes do dataset **Stanford IMDB** (via Hugging Face), preparando a base para análises de sentimentos e recomendações futuras.

---

### Escopo Atual
| Etapa | Entrega | Status |
|:--|:--|:--:|
| 01 | Início do projeto | ✅ Concluída |
| 02 | Coleta de dados | ✅ Concluída |
| 03 | Pré-processamento de dados | ✅ Refatorada (Google Colab) |
| 04 | Aprendizagem de máquina | ✅ Concluída |
| 05 | Análise dos resultados | ✅ Concluída |
| 06 | Otimizações | ⏳ Planejada |

---

### Arquitetura de Referência


```
Hugging Face (IMDB) ──► Google Colab (Script de Coleta)
                              │
                              ▼
                         Arquivos CSV locais
                              │
                              ▼
                        Google Colab (Script de Processamento PySpark)
                              │
                              ▼
                         Featurizações (HTF, TF-IDF, Word2Vec)
                              │
                              ▼
                        Treinamento de Modelos (LR, SVM)
                              │
                              ▼
                         Análise de Resultados & Métricas
```                             


### Stack Técnica
- **Google Colab / Python 3**
- **PySpark 3.5.1** (processamento distribuído)
- **Pandas** (manipulação de dados)
- **NumPy** (computação científica)
- **scikit-learn** (machine learning)
- **Hugging Face Datasets** (carregamento de dados)
- **NLTK** (pré-processamento de texto)
- **Java OpenJDK 17** (runtime para Spark)

---

### Execução da Etapa 02 – Coleta de Dados
Notebook: [`coleta_dados.ipynb`](./coleta_dados.ipynb)

1. Acesse o Google Colab e importe o notebook.  
2. Execute as células sequencialmente:  
   - Carrega o dataset **stanfordnlp/imdb** via Hugging Face.  
   - Converte e salva os dados em formato **Parquet** ou **CSV** localmente.  
3. Os dados resultantes servirão de entrada para o notebook da etapa 03.

---

### Execução da Etapa 03 – Pré-processamento de Dados
Notebook: [`preprocessamento.ipynb`](./preprocessamento.ipynb)

1. Configura ambiente Spark + Java no Google Colab.  
2. Carrega os dados CSV gerados pela etapa 02.  
3. Realiza limpeza textual (remoção de HTML, normalização, tokenização).  
4. Implementa **três tipos de featurização:**
   - **HTF (Hashing TF):** Features baseadas em hashing para unigramas
   - **TF-IDF:** Combinação de unigramas + bigramas com vocabulário controlado  
   - **Word2Vec:** Embeddings vetoriais com escalonamento MinMax
5. Exporta datasets featurizados em formato **Parquet**.

---

### Execução da Etapa 04 – Aprendizado de Máquina
Notebook: [`aprendizado_maquina.ipynb`](./aprendizado_maquina.ipynb)

1. Carrega as três featurizações geradas na etapa 03.  
2. Implementa e treina **dois algoritmos de classificação:**
   - **Logistic Regression** com validação cruzada e grid search
   - **Linear SVC** (Support Vector Machine) com tratamento para multiclasse
3. Compara performance entre todas as combinações modelo+featurização.  
4. Identifica automaticamente a **melhor configuração** baseada na acurácia.

---

### Execução da Etapa 05 – Análise de Resultados
Notebook: [`analise_resultados.ipynb`](./analise_resultados.ipynb)

1. Utiliza a melhor featurização identificada (TF-IDF).  
2. Gera métricas detalhadas de avaliação:
   - **Acurácia** e **Taxa de erro**
   - **F1-score** para balanceamento de classes
   - **Matriz de confusão** para análise de erros
3. Compara performance final entre Logistic Regression e Linear SVC.

---

### Roadmap Próximo
- **Etapa 06 – Otimizações:** fine-tuning de hiperparâmetros e feature engineering avançado  
- **Etapa 07 – Deploy:** disponibilização via API REST ou interface web  

---

### Equipe
- **Andressa Cristina Chaves De Oliveira**  
- **Ravi Ferreira Pellizzi**  
- **Rafael Evangelista Oliveira**  
- **Calebe Stoffel de Castro Moura**  
- **Luana Patricia Gonçalves Machado**  
- **Igor Vinicius da Silva Nascimento**

**Orientação:** Cristiano Geraldo Teixeira Silva

---

