# Eixo 5 | Arquitetura de Dados em Nuvem
## Pipeline de Sentimentos IMDB

### Visão Geral
Pipeline em Databricks/Spark que captura avaliações públicas de filmes (`stanfordnlp/imdb`), persiste a camada Bronze em Parquet no Amazon S3 e prepara a base para análises de sentimentos e recomendações futuras. Projeto desenvolvido na graduação em Tecnologia em Banco de Dados (2025/2).

### Escopo Atual (Etapa 02)
- Infraestrutura provisionada via Terraform para Databricks e S3.
- Job de ingestão executa leitura no Hugging Face e grava a camada Bronze em Parquet.

### Roadmap
| Etapa | Entrega | Status |
| --- | --- | --- |
| 01 | Arquitetura e planejamento | ✅ Concluída |
| 02 | Coleta e ingestão (Bronze) | ✅ Concluída |
| 03 | Limpeza e processamento (Silver/Gold) | ⏳ Planejada |
| 04 | Insights, ML e análise de resultados | ⏳ Planejada |

### Arquitetura de Referência
```
Hugging Face (IMDB) ──► Databricks Jobs (Spark)
                              │
                              ▼
                         Amazon S3 (Parquet)
                              └─► Camadas Silver/Gold & análises (próximo)
```

### Stack
- Databricks (jobs, clusters, secrets)
- Apache Spark
- Amazon S3
- Terraform
- Docker (execução opcional das ferramentas de IaC)

### Execução da Etapa 02
1. **Pré-requisitos**
   - Conta AWS com Access/Secret Key válidas para S3.
   - Workspace Databricks com PAT ativo.
   - Terraform `>= 1.5` local ou uso do `Dockerfile` deste repositório.
   - Databricks CLI autenticado.

2. **Segredos no Databricks**
   ```bash
   databricks configure --token
   databricks secrets create-scope --scope aws
   databricks secrets put --scope aws --key aws_access_key_id
   databricks secrets put --scope aws --key aws_secret_access_key
   ```

3. **Provisionamento**
   Ajuste os arquivos `*.auto.tfvars` (bucket, região, cluster). Em seguida:
   ```bash
   terraform init
   terraform apply
   ```
   Ou, com Docker:
   ```bash
   docker build -t tf-db-ingest .
   docker run --rm -it \
     -v "$PWD":/iac -w /iac \
     -e DATABRICKS_HOST="$DATABRICKS_HOST" \
     -e DATABRICKS_TOKEN="$DATABRICKS_TOKEN" \
     tf-db-ingest sh -lc "terraform init && terraform apply -auto-approve"
   ```

4. **Ingestão**
   - Localize o job criado no workspace Databricks.
   - Informe os widgets:
     - `dataset_name`: `stanfordnlp/imdb`
     - `s3_path`: `s3://<bucket>/hf/imdb_parquet/`
   - Execute `Run now`.

5. **Validação**
   ```bash
   aws s3 ls s3://<bucket>/hf/imdb_parquet/ --recursive
   ```
   ```python
   spark.read.parquet("s3a://<bucket>/hf/imdb_parquet/").show(5)
   ```

### Equipe
- Andressa Cristina Chaves De Oliveira
- Ravi Ferreira Pellizzi
- Rafael Evangelista Oliveira
- Calebe Stoffel de Castro Moura
- Luana Patricia Gonçalves Machado
- Igor Vinicius da Silva Nascimento

### Orientação
- Cristiano Geraldo Teixeira Silva
