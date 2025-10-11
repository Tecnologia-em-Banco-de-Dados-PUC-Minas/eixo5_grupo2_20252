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
| 01 | Arquitetura e planejamento | ✅ Concluída |
| 02 | Coleta e ingestão (Bronze) | ✅ Refatorada (Google Colab) |
| 03 | Limpeza e processamento (Silver/Gold) | ✅ Concluída |
| 04 | Insights, ML e análise de resultados | ⏳ Planejada |

---

### Arquitetura de Referência


```
Hugging Face (IMDB) ──► Google Colab (Script de Coleta)
                              │
                              ▼
                         Arquivos Parquet/CSV locais
                              │
                             ▼
                        Google Colab (Script de Processamento PySpark)
                             └─► Camadas Silver/Gold & análises (atuais e futuras)
```                             


### Stack Técnica
- **Google Colab / Python 3**
- **Pandas**
- **NumPy**
- **scikit-learn**
- **Hugging Face Datasets**
- **Matplotlib / Seaborn**
- **NLTK (para pré-processamento de texto)**

---

### Execução da Etapa 02 – Coleta de Dados
Notebook: [`coleta_dados.ipynb`](./coleta_dados.ipynb)

1. Acesse o Google Colab e importe o notebook.  
2. Execute as células sequencialmente:  
   - Carrega o dataset **stanfordnlp/imdb** via Hugging Face.  
   - Converte e salva os dados em formato **Parquet** ou **CSV** localmente.  
3. Os dados resultantes servirão de entrada para o notebook da etapa 03.

---

### Execução da Etapa 03 – Processamento e Análise
Notebook: [`processamento.ipynb`](./processamento.ipynb)

1. Carregue os arquivos gerados pela etapa 02.  
2. Realiza limpeza, tokenização e normalização textual.  
3. Gera métricas e visualizações iniciais de distribuição de sentimentos.  
4. Exporta a base tratada nas camadas **Silver/Gold**.

---

### Roadmap Próximo
- **Etapa 04 – Modelagem de Machine Learning:** criação de modelos de classificação de sentimentos.  
- **Etapa 05 – Dashboard / API:** disponibilização dos resultados via Streamlit ou FastAPI.  

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

