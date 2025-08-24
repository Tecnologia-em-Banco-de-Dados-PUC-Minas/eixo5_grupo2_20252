## 📌 Definição do Problema

### Problema
No cenário atual de plataformas de streaming e entretenimento digital, os usuários frequentemente enfrentam dificuldade em escolher conteúdos relevantes diante da grande oferta de filmes disponíveis. Isso pode gerar frustração, perda de tempo e, em alguns casos, até o cancelamento de serviços por falta de engajamento.


---

### Contexto
O projeto utilizará **datasets públicos disponíveis no Kaggle**, tais como:
- **MovieLens Dataset (Kaggle)**: Contém informações de usuários, filmes e ratings.
  - **[MovieLens 20M Dataset (Kaggle)](https://www.kaggle.com/datasets/grouplens/movielens-20m-dataset?utm_source=chatgpt.com)** – Contém 20 milhões de ratings de filmes por usuários, ideal para construção do modelo de recomendação.  
  - **[MovieLens 100K Dataset (Kaggle)](https://www.kaggle.com/datasets/prajitdatta/movielens-100k-dataset?utm_source=chatgpt.com)** – Versão menor, útil para testes rápidos.  
  - **[MovieLens 1M Dataset (Kaggle)](https://www.kaggle.com/datasets/odedgolden/movielens-1m-dataset?utm_source=chatgpt.com)** – Versão intermediária entre 100K e 20M.  
  - **[MovieLens Latest Datasets (Oficial GroupLens)](https://grouplens.org/datasets/movielens/?utm_source=chatgpt.com)** – Fonte oficial, com versões atualizadas e estáveis.  

- **IMDB Reviews Dataset (Kaggle)**: Base de dados com críticas de filmes em formato textual, útil para a análise de sentimentos.
  - **[IMDB 50K Movie Reviews Dataset (Kaggle)](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews?utm_source=chatgpt.com)** – Base com 50 mil críticas de filmes, balanceadas entre positivas e negativas.  
  - **[Stanford Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/?utm_source=chatgpt.com)** – Dataset acadêmico com 25 mil críticas positivas e 25 mil negativas, amplamente utilizado em pesquisas de PLN.  

---
**Viabilidade:**
- Os datasets são amplamente utilizados em pesquisas acadêmicas e projetos de machine learning.
- Possuem volume de dados suficiente para simular um ambiente real de recomendação.
- A integração entre dados estruturados (ratings) e não estruturados (reviews de texto) permite enriquecer o modelo com múltiplas fontes de informação.

---

### Objetivos
**Objetivos iniciais:**
1. Explorar e preparar os dados de filmes, usuários e avaliações.
2. Implementar técnicas de **Processamento de Linguagem Natural (PLN)** para análise de sentimentos em críticas de filmes.
3. Construir um modelo de recomendação que combine:
   - Dados quantitativos (ratings).
   - Dados qualitativos (sentimentos extraídos de reviews).
4. Criar uma arquitetura de dados em nuvem escalável para ingestão, transformação e análise dos dados.

**Resultados esperados:**
- Um sistema de recomendação de filmes que sugira conteúdos de forma personalizada e com maior precisão.
- Relatórios e dashboards que permitam avaliar métricas de desempenho do modelo (ex.: precisão, recall, cobertura).
- Um pipeline de dados rastreável, sustentável e seguro para futuras expansões.
