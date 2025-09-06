# Sistema de Processamento de Dados de Filmes - PostgreSQL

## Visão Geral

Este sistema processa dados de filmes do MovieLens e os carrega em um banco de dados PostgreSQL.

## Pré-requisitos

### Software Necessário

1. **PostgreSQL 12 ou superior**
   - Instale o PostgreSQL em seu sistema
   - Certifique-se de que o serviço está rodando

2. **Python 3.8 ou superior**

3. **Bibliotecas Python necessárias:**
   ```bash
   pip install pandas psycopg2-binary sqlalchemy configparser
   ```

### Arquivos de Dados (CSV)

Os arquivos devem estar na pasta `Bases/`:
- `movie.csv` - Informações dos filmes
- `rating.csv` - Avaliações dos usuários
- `tag.csv` - Tags criadas pelos usuários
- `link.csv` - Links para IMDB e TMDB
- `genome_tags.csv` - Tags do genoma
- `genome_scores.csv` - Scores de relevância das tags

## Configuração

### 1. Criar o Banco de Dados

Conecte-se ao PostgreSQL como superusuário e execute:

```sql
-- Criar o banco de dados
CREATE DATABASE movie_database;

-- Conectar ao banco criado
\c movie_database
```

### 2. Configurar Arquivo de Conexão

Edite o arquivo `database_config.ini` com suas credenciais:

```ini
[postgresql]
host = localhost
port = 5432
database = movie_database
user = postgres
password = sua_senha
```

**Importante:** Mantenha este arquivo seguro e não o compartilhe em repositórios públicos.

## Como Executar

### 1. Preparação do Ambiente

```bash
# Instalar dependências
pip install pandas psycopg2-binary sqlalchemy configparser
```

### 2. Executar o Processamento

```bash
# Executar o script principal
python movie_data_processor_postgresql.py
```

### 3. Monitorar o Progresso

O script fornece logging detalhado mostrando:
- Validação de arquivos
- Teste de conexão
- Criação/remoção de tabelas
- Progresso do carregamento por chunks
- Estatísticas finais

## Estrutura do Banco

O script cria as seguintes tabelas:

- **movies**: Informações dos filmes (movie_id, title, genres, created_at)
- **movie_links**: Links para IMDB e TMDB (movie_id, imdb_id, tmdb_id)
- **users**: Usuários únicos extraídos dos ratings e tags
- **ratings**: Avaliações dos usuários (user_id, movie_id, rating, timestamp)
- **user_tags**: Tags criadas pelos usuários (user_id, movie_id, tag, timestamp)
- **genome_tags**: Tags do genoma (tag_id, tag_name)
- **genome_scores**: Scores de relevância das tags (movie_id, tag_id, relevance)

## Funcionalidades do Sistema

### Recursos Básicos
- ✅ Validação de arquivos antes do processamento
- ✅ Carregamento otimizado com chunks para grandes volumes
- ✅ Análise de relacionamentos e estatísticas
- ✅ Logging detalhado do processo
- ✅ Tratamento de erros robusto
- ✅ Conversão inteligente de timestamps
- ✅ Limpeza automática de tabelas existentes

### Recursos Avançados do PostgreSQL

#### Views Materializadas

O sistema cria views materializadas para consultas frequentes:


#### Busca Textual Avançada


```sql
-- Buscar filmes por título
SELECT movie_id, title, 
       ts_rank(to_tsvector('english', title), query) as rank
FROM movies, to_tsquery('english', 'toy & story') query
WHERE to_tsvector('english', title) @@ query
ORDER BY rank DESC;

-- Buscar por gênero
SELECT * FROM movies 
WHERE to_tsvector('english', genres) @@ to_tsquery('english', 'Comedy');

-- Buscar tags com similaridade
SELECT tag FROM user_tags 
WHERE tag like '%action%' 
ORDER BY tag DESC;
```

## Consultas de Exemplo

### Análises Básicas

```sql
-- Top 10 filmes mais bem avaliados
SELECT 
    m.title,
    AVG(r.rating) as avg_rating,
    COUNT(r.rating) as rating_count
FROM movies m
INNER JOIN ratings r ON m.movie_id = r.movie_id
GROUP BY m.movie_id, m.title
HAVING COUNT(r.rating) >= 50
ORDER BY avg_rating DESC
LIMIT 10;

-- Análise por gênero
SELECT 
    unnest(string_to_array(genres, '|')) as genre,
    COUNT(*) as movie_count,
    AVG(rating) as avg_rating
FROM movies m
INNER JOIN ratings r ON m.movie_id = r.movie_id
GROUP BY unnest(string_to_array(genres, '|'))
ORDER BY movie_count DESC;

-- Tags mais populares
SELECT 
    tag,
    COUNT(*) as usage_count
FROM user_tags
GROUP BY tag
ORDER BY usage_count DESC
LIMIT 20;
```

### Análises Avançadas

```sql
-- Análise temporal de avaliações
SELECT 
    DATE_TRUNC('month', timestamp) as month,
    COUNT(*) as ratings_count,
    AVG(rating) as avg_rating
FROM ratings
WHERE timestamp >= '2020-01-01'
GROUP BY DATE_TRUNC('month', timestamp)
ORDER BY month;

-- Filmes com maior variação nas avaliações
SELECT 
    m.title,
    COUNT(r.rating) as rating_count,
    AVG(r.rating) as avg_rating,
    STDDEV(r.rating) as rating_stddev
FROM movies m
INNER JOIN ratings r ON m.movie_id = r.movie_id
GROUP BY m.movie_id, m.title
HAVING COUNT(r.rating) >= 50
ORDER BY STDDEV(r.rating) DESC
LIMIT 20;
```

### Recomendações e Similaridade

```sql
-- Filmes similares baseado em genome tags
SELECT * FROM get_similar_movies(1);

-- Recomendações para usuário específico
SELECT * FROM get_user_recommendations(123);

-- Usuários com gostos similares
WITH user_correlations AS (
    SELECT 
        r1.user_id as user1,
        r2.user_id as user2,
        CORR(r1.rating, r2.rating) as correlation,
        COUNT(*) as common_movies
    FROM ratings r1
    INNER JOIN ratings r2 ON r1.movie_id = r2.movie_id
    WHERE r1.user_id < r2.user_id
    GROUP BY r1.user_id, r2.user_id
    HAVING COUNT(*) >= 20
)
SELECT * FROM user_correlations 
WHERE correlation > 0.7
ORDER BY correlation DESC;
```

### Problemas Comuns

#### 1. Erro de Conexão
```
psycopg2.OperationalError: could not connect to server
```

**Soluções:**
- Verificar se PostgreSQL está rodando
- Confirmar host, porta e credenciais
- Verificar configurações de firewall
- Checar arquivo pg_hba.conf para permissões

#### 2. Erro de Permissões
```
permission denied for table movies
```

**Soluções:**
- Conceder privilégios adequados ao usuário
- Verificar ownership das tabelas
- Usar superusuário se necessário

#### 3. Problemas de Memória
```
out of memory
```

**Soluções:**
- Reduzir tamanho dos chunks no script
- Aumentar shared_buffers no PostgreSQL
- Configurar work_mem adequadamente

#### 4. Performance Lenta
**Soluções:**
- Verificar se índices foram criados
- Executar ANALYZE nas tabelas
- Configurar parâmetros de performance
- Considerar particionamento para tabelas muito grandes

### Configurações Recomendadas do PostgreSQL

Para melhor performance com grandes volumes:

```sql
-- postgresql.conf
shared_buffers = 256MB
effective_cache_size = 1GB
work_mem = 4MB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
```

## Arquivos do Sistema

### Arquivos Principais
- **movie_data_processor_postgresql.py** - Script principal otimizado
- **emergency_clear.py** - Script de emergência para limpeza
- **database_config.ini** - Configurações de conexão
- **README.md** - Esta documentação

### Scripts de Emergência

Se o processamento travar, use:
```bash
python emergency_clear.py
```
