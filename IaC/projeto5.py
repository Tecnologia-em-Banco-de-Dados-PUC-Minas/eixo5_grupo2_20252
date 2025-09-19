%pip install -q datasets==2.20.0

# COMMAND ----------
import os
from datasets import load_dataset

# Parâmetros mínimos
dbutils.widgets.text("dataset_name", "stanfordnlp/imdb", "Dataset")
dbutils.widgets.text("s3_path", "", "S3 destination path")

dataset_name = dbutils.widgets.get("dataset_name").strip() or "stanfordnlp/imdb"
s3_path = dbutils.widgets.get("s3_path").strip()

if not s3_path:
    raise ValueError("Defina 's3_path' via widget.")

# Credenciais AWS via Databricks Secrets (escopo: aws)
aws_access_key_id = dbutils.secrets.get("aws", "aws_access_key_id")
aws_secret_access_key = dbutils.secrets.get("aws", "aws_secret_access_key")

# Configura S3A para usar as chaves diretamente
spark.conf.set("fs.s3a.aws.credentials.provider", "org.apache.hadoop.fs.s3a.SimpleAWSCredentialsProvider")
spark.conf.set("fs.s3a.access.key", aws_access_key_id)
spark.conf.set("fs.s3a.secret.key", aws_secret_access_key)

# Carrega o dataset do Hugging Face
ds = load_dataset(dataset_name)

# Une todos os splits em um único DataFrame Spark e grava em Parquet
sdf_merged = None
for _, split in ds.items():
    pdf = split.to_pandas()
    sdf_split = spark.createDataFrame(pdf)
    sdf_merged = sdf_split if sdf_merged is None else sdf_merged.unionByName(sdf_split, allowMissingColumns=True)

sdf_merged.write.mode("overwrite").parquet(s3_path)
