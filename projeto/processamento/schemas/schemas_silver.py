from pyspark.sql.types import StructType, StructField, StringType, TimestampType, IntegerType, DoubleType

genome_scores = StructType([ # que contém dados de relevância da tag de filme
    StructField('id_filme', IntegerType(), nullable=True),
    StructField('identificacao_tag', IntegerType(), nullable=True),
    StructField('relevancia', DoubleType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])

genome_tags = StructType([ # contém descrições de tags
    StructField('identificacao_tag', IntegerType(), nullable=True),
    StructField('etiqueta', StringType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])

link = StructType([ # contém identificadores que podem ser usados para vincular a outras fontes
    StructField('id_filme', IntegerType(), nullable=True),
    StructField('imdbId', IntegerType(), nullable=True),
    StructField('tmdbId', IntegerType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])

movie = StructType([ # contém informações sobre o filme
    StructField('id_filme', IntegerType(), nullable=True),
    StructField('imdbId', StringType(), nullable=True),
    StructField('tmdbId', StringType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])

rating = StructType([ # contém classificações de filmes pelos usuários
    StructField('id_usuario', IntegerType(), nullable=True),
    StructField('id_filme', IntegerType(), nullable=True),
    StructField('classificacao', DoubleType(), nullable=True),
    StructField('timestamp', TimestampType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])

tag = StructType([ # contém tags aplicadas a filmes pelos usuários
    StructField('id_usuario', IntegerType(), nullable=True),
    StructField('id_filme', IntegerType(), nullable=True),
    StructField('etiqueta', StringType(), nullable=True),
    StructField('timestamp', TimestampType(), nullable=True),
    StructField('data_ingestao', TimestampType(), nullable=False)
])
