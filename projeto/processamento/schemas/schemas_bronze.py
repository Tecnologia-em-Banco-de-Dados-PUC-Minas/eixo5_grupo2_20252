from pyspark.sql.types import StructType, StructField, StringType, TimestampType

genome_scores = StructType([ # que contém dados de relevância da tag de filme
    StructField('id_filme', StringType(), nullable=True),
    StructField('identificacao_tag', StringType(), nullable=True),
    StructField('relevancia', StringType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])

genome_tags = StructType([ # contém descrições de tags
    StructField('identificacao_tag', StringType(), nullable=True),
    StructField('etiqueta', StringType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])

link = StructType([ # contém identificadores que podem ser usados para vincular a outras fontes
    StructField('id_filme', StringType(), nullable=True),
    StructField('imdbId', StringType(), nullable=True),
    StructField('tmdbId', StringType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])

movie = StructType([ # contém informações sobre o filme
    StructField('id_filme', StringType(), nullable=True),
    StructField('imdbId', StringType(), nullable=True),
    StructField('tmdbId', StringType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])

rating = StructType([ # contém classificações de filmes pelos usuários
    StructField('id_usuario', StringType(), nullable=True),
    StructField('id_filme', StringType(), nullable=True),
    StructField('classificacao', StringType(), nullable=True),
    StructField('timestamp', StringType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])

tag = StructType([ # contém tags aplicadas a filmes pelos usuários
    StructField('id_usuario', StringType(), nullable=True),
    StructField('id_filme', StringType(), nullable=True),
    StructField('etiqueta', StringType(), nullable=True),
    StructField('timestamp', StringType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])
