#!/usr/bin/env python3
"""
Script para processar dados de filmes e salvar no PostgreSQL
Versão adaptada para banco de dados PostgreSQL
"""

import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import os
from datetime import datetime
import logging
import gc
from sqlalchemy import create_engine
import configparser

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MovieDataProcessorPostgreSQL:
    def __init__(self, data_dir="Bases", config_file="database_config.ini", clear_tables=False):
        """
        Inicializa o processador de dados de filmes para PostgreSQL
        
        Args:
            data_dir (str): Diretório contendo os arquivos CSV
            config_file (str): Arquivo de configuração do banco
            clear_tables (bool): Se True, limpa as tabelas antes de inserir dados
        """
        self.data_dir = data_dir
        self.config_file = config_file
        self.clear_tables = clear_tables
        self.connection = None
        self.engine = None
        
        # Carregar configurações do banco
        self.db_config = self.load_database_config()
        
        # Definir caminhos dos arquivos
        self.files = {
            'movies': os.path.join(data_dir, 'movie.csv'),
            'ratings': os.path.join(data_dir, 'rating.csv'),
            'tags': os.path.join(data_dir, 'tag.csv'),
            'links': os.path.join(data_dir, 'link.csv'),
            'genome_tags': os.path.join(data_dir, 'genome_tags.csv'),
            'genome_scores': os.path.join(data_dir, 'genome_scores.csv')
        }
        
    def load_database_config(self):
        """Carrega configurações do banco de dados"""
        config = configparser.ConfigParser()
        
        # Configurações padrão
        default_config = {
            'host': 'localhost',
            'port': '5432',
            'database': 'movie_database',
            'user': 'postgres',
            'password': 'password'
        }
        
        if os.path.exists(self.config_file):
            try:
                # Tentar ler com encoding UTF-8 primeiro
                config.read(self.config_file, encoding='utf-8')
                if 'postgresql' in config:
                    return dict(config['postgresql'])
            except UnicodeDecodeError:
                try:
                    # Se falhar, tentar com encoding latin-1
                    config.read(self.config_file, encoding='latin-1')
                    if 'postgresql' in config:
                        return dict(config['postgresql'])
                except Exception as e:
                    logger.warning(f"Erro ao ler arquivo de configuração: {e}")
                    logger.info("Usando configurações padrão")
        
        # Se não existe arquivo de config ou houve erro, criar um com valores padrão
        self.create_default_config(default_config)
        return default_config
        
    def create_default_config(self, config):
        """Cria arquivo de configuração padrão"""
        config_parser = configparser.ConfigParser()
        config_parser['postgresql'] = config
        
        try:
            with open(self.config_file, 'w', encoding='utf-8') as f:
                config_parser.write(f)
                
            logger.info(f"Arquivo de configuração criado: {self.config_file}")
            logger.info("Por favor, edite o arquivo com suas credenciais do PostgreSQL")
        except Exception as e:
            logger.error(f"Erro ao criar arquivo de configuração: {e}")
            logger.info("Usando configurações padrão em memória")
        
    def connect_database(self):
        """Conecta ao banco de dados PostgreSQL"""
        try:
            # Validar e limpar configurações
            host = str(self.db_config['host']).strip()
            port = str(self.db_config['port']).strip()
            database = str(self.db_config['database']).strip()
            user = str(self.db_config['user']).strip()
            password = str(self.db_config['password']).strip()
            
            logger.info(f"Tentando conectar ao PostgreSQL: {host}:{port}")
            logger.info(f"Banco: {database}, Usuário: {user}")
            
            # Conexão direta com psycopg2
            self.connection = psycopg2.connect(
                host=host,
                port=port,
                database=database,
                user=user,
                password=password
            )
            self.connection.autocommit = False
            
            # Engine do SQLAlchemy para pandas
            connection_string = (
                f"postgresql://{user}:{password}"
                f"@{host}:{port}/{database}"
            )
            self.engine = create_engine(connection_string)
            
            logger.info(f"Conectado ao PostgreSQL: {host}:{port}")
            
        except UnicodeDecodeError as e:
            logger.error(f"Erro de codificação nas configurações: {e}")
            logger.error("Verifique se há caracteres especiais no arquivo database_config.ini")
            logger.error("Tente recriar o arquivo de configuração")
            raise
        except Exception as e:
            logger.error(f"Erro ao conectar ao banco: {e}")
            logger.error("Verifique as configurações no arquivo database_config.ini")
            logger.error("Certifique-se de que o PostgreSQL está rodando e acessível")
            raise
            
    def create_tables(self):
        """Cria as tabelas no PostgreSQL sem PK, FK ou restrições"""
        cursor = self.connection.cursor()
        try:
            # Primeiro, remover todas as tabelas existentes com CASCADE
            logger.info("Removendo tabelas existentes...")
            tables_to_drop = [
                'genome_scores',
                'user_tags', 
                'ratings',
                'users',
                'genome_tags',
                'movie_links',
                'movies'
            ]
            
            for table in tables_to_drop:
                try:
                    cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
                    logger.info(f"Tabela {table} removida")
                except Exception as e:
                    logger.warning(f"Erro ao remover tabela {table}: {e}")
            
            # Criar tabelas sem restrições
            cursor.execute("""
                CREATE TABLE movies (
                    movie_id INTEGER,
                    title TEXT,
                    genres TEXT,
                    created_at TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE movie_links (
                    movie_id INTEGER,
                    imdb_id TEXT,
                    tmdb_id TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE users (
                    user_id INTEGER,
                    created_at TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE ratings (
                    id SERIAL,
                    user_id INTEGER,
                    movie_id INTEGER,
                    rating DECIMAL(2,1),
                    timestamp TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE user_tags (
                    id SERIAL,
                    user_id INTEGER,
                    movie_id INTEGER,
                    tag TEXT,
                    timestamp TIMESTAMP
                )
            """)
            cursor.execute("""
                CREATE TABLE genome_tags (
                    tag_id INTEGER,
                    tag_name TEXT
                )
            """)
            cursor.execute("""
                CREATE TABLE genome_scores (
                    id SERIAL,
                    movie_id INTEGER,
                    tag_id INTEGER,
                    relevance DECIMAL(10,8)
                )
            """)
            self.connection.commit()
            logger.info("Tabelas criadas sem PK/FK/índices no PostgreSQL")
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Erro ao criar tabelas: {e}")
            raise

    def clear_tables_data(self):
        """Limpa todos os dados das tabelas de forma otimizada"""
        cursor = self.connection.cursor()
        try:
            # Desabilitar autocommit para melhor performance
            self.connection.autocommit = False
            
            # Ordem de limpeza considerando dependências
            tables_to_clear = [
                'genome_scores',
                'user_tags', 
                'ratings',
                'users',
                'genome_tags',
                'movie_links',
                'movies'
            ]
            
            # Verificar se há dados para limpar
            total_records = 0
            for table in tables_to_clear:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                total_records += count
                logger.info(f"Tabela {table}: {count} registros")
            
            if total_records == 0:
                logger.info("Todas as tabelas já estão vazias, pulando limpeza")
                return
            
            logger.info(f"Total de registros para limpar: {total_records}")
            
            for table in tables_to_clear:
                logger.info(f"Limpando tabela {table}...")
                
                # Usar TRUNCATE que é muito mais rápido que DELETE
                try:
                    cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
                    logger.info(f"Tabela {table} truncada com sucesso")
                except Exception as truncate_error:
                    # Se TRUNCATE falhar, usar DELETE como fallback
                    logger.warning(f"TRUNCATE falhou para {table}, usando DELETE: {truncate_error}")
                    cursor.execute(f"DELETE FROM {table}")
                    logger.info(f"Tabela {table} limpa com DELETE")
            
            self.connection.commit()
            logger.info("Todas as tabelas foram limpas com sucesso")
            
        except Exception as e:
            self.connection.rollback()
            logger.error(f"Erro ao limpar tabelas: {e}")
            raise

    def load_movies(self):
        """Carrega dados de movies.csv para a tabela movies"""
        df = pd.read_csv(self.files['movies'])
        df['created_at'] = datetime.now()
        df = df.rename(columns={'movieId': 'movie_id'})
        df = df[['movie_id', 'title', 'genres', 'created_at']]
        
        # Tabelas sempre vazias após recriação, usar append
        df.to_sql('movies', self.engine, if_exists='append', index=False)
        logger.info(f"{len(df)} filmes inseridos na tabela movies")

    def load_links(self):
        """Carrega dados de link.csv para a tabela movie_links"""
        df = pd.read_csv(self.files['links'])
        df = df.rename(columns={'movieId': 'movie_id'})
        df = df[['movie_id', 'imdbId', 'tmdbId']]
        df = df.rename(columns={'imdbId': 'imdb_id', 'tmdbId': 'tmdb_id'})
        
        df.to_sql('movie_links', self.engine, if_exists='append', index=False)
        logger.info(f"{len(df)} links inseridos na tabela movie_links")

    def load_genome_tags(self):
        """Carrega dados de genome_tags.csv para a tabela genome_tags"""
        df = pd.read_csv(self.files['genome_tags'])
        df = df.rename(columns={'tagId': 'tag_id'})
        df = df[['tag_id', 'tag']]
        df = df.rename(columns={'tag': 'tag_name'})
        
        df.to_sql('genome_tags', self.engine, if_exists='append', index=False)
        logger.info(f"{len(df)} tags do genoma inseridas na tabela genome_tags")

    def extract_users(self):
        """Extrai usuários únicos dos dados de ratings e tags"""
        users_data = []
        
        # Extrair usuários dos ratings
        ratings_df = pd.read_csv(self.files['ratings'])
        unique_users_ratings = ratings_df['userId'].unique()
        
        # Extrair usuários das tags
        tags_df = pd.read_csv(self.files['tags'])
        unique_users_tags = tags_df['userId'].unique()
        
        # Combinar usuários únicos
        all_users = set(unique_users_ratings) | set(unique_users_tags)
        
        # Criar DataFrame de usuários
        users_df = pd.DataFrame({
            'user_id': list(all_users),
            'created_at': datetime.now()
        })
        
        users_df.to_sql('users', self.engine, if_exists='append', index=False)
        logger.info(f"{len(users_df)} usuários inseridos na tabela users")

    def load_ratings(self):
        """Carrega dados de rating.csv para a tabela ratings"""
        df = pd.read_csv(self.files['ratings'])
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
        
        # Converter timestamp - tentar diferentes formatos
        try:
            # Primeiro, tentar como timestamp Unix
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        except ValueError:
            try:
                # Se falhar, tentar como string de data/hora
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                logger.warning(f"Erro ao converter timestamp: {e}")
                # Se tudo falhar, usar data atual
                df['timestamp'] = pd.Timestamp.now()
        
        df = df[['user_id', 'movie_id', 'rating', 'timestamp']]
        
        df.to_sql('ratings', self.engine, if_exists='append', index=False, chunksize=10000)
        logger.info(f"{len(df)} avaliações inseridas na tabela ratings")

    def load_user_tags(self):
        """Carrega dados de tag.csv para a tabela user_tags"""
        df = pd.read_csv(self.files['tags'])
        df = df.rename(columns={'userId': 'user_id', 'movieId': 'movie_id'})
        
        # Converter timestamp - tentar diferentes formatos
        try:
            # Primeiro, tentar como timestamp Unix
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        except ValueError:
            try:
                # Se falhar, tentar como string de data/hora
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            except Exception as e:
                logger.warning(f"Erro ao converter timestamp: {e}")
                # Se tudo falhar, usar data atual
                df['timestamp'] = pd.Timestamp.now()
        
        df = df[['user_id', 'movie_id', 'tag', 'timestamp']]
        
        df.to_sql('user_tags', self.engine, if_exists='append', index=False, chunksize=10000)
        logger.info(f"{len(df)} tags de usuário inseridas na tabela user_tags")

    def load_genome_scores(self):
        """Carrega dados de genome_scores.csv para a tabela genome_scores"""
        df = pd.read_csv(self.files['genome_scores'])
        df = df.rename(columns={'movieId': 'movie_id', 'tagId': 'tag_id'})
        df = df[['movie_id', 'tag_id', 'relevance']]
        
        df.to_sql('genome_scores', self.engine, if_exists='append', index=False, chunksize=10000)
        logger.info(f"{len(df)} scores do genoma inseridos na tabela genome_scores")

    def analyze_relationships(self):
        """Analisa relacionamentos e retorna estatísticas"""
        cursor = self.connection.cursor()
        stats = {}
        
        try:
            # Contar registros em cada tabela
            tables = ['movies', 'movie_links', 'users', 'ratings', 'user_tags', 'genome_tags', 'genome_scores']
            
            for table in tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                stats[f"Total {table}"] = count
            
            # Análise de relacionamentos
            cursor.execute("""
                SELECT COUNT(DISTINCT r.user_id) 
                FROM ratings r 
                INNER JOIN movies m ON r.movie_id = m.movie_id
            """)
            stats["Usuários com avaliações válidas"] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT COUNT(DISTINCT r.movie_id) 
                FROM ratings r 
                INNER JOIN movies m ON r.movie_id = m.movie_id
            """)
            stats["Filmes com avaliações"] = cursor.fetchone()[0]
            
            cursor.execute("""
                SELECT AVG(rating) 
                FROM ratings
            """)
            avg_rating = cursor.fetchone()[0]
            stats["Avaliação média"] = round(avg_rating, 2) if avg_rating else 0
            
            cursor.execute("""
                SELECT COUNT(DISTINCT tag) 
                FROM user_tags
            """)
            stats["Tags únicas"] = cursor.fetchone()[0]
            
        except Exception as e:
            logger.error(f"Erro na análise de relacionamentos: {e}")
            stats["Erro na análise"] = str(e)
        
        return stats

    def validate_files(self):
        """Valida se todos os arquivos necessários existem"""
        missing_files = []
        for name, path in self.files.items():
            if not os.path.exists(path):
                missing_files.append(f"{name}: {path}")
        
        if missing_files:
            error_msg = "Arquivos não encontrados:\n" + "\n".join(missing_files)
            logger.error(error_msg)
            raise FileNotFoundError(error_msg)
        
        logger.info("Todos os arquivos de dados foram encontrados")

    def test_connection(self):
        """Testa a conexão com o banco de dados"""
        try:
            logger.info("Testando conexão com PostgreSQL...")
            test_conn = psycopg2.connect(
                host=str(self.db_config['host']).strip(),
                port=str(self.db_config['port']).strip(),
                database=str(self.db_config['database']).strip(),
                user=str(self.db_config['user']).strip(),
                password=str(self.db_config['password']).strip()
            )
            test_conn.close()
            logger.info("Conexão de teste bem-sucedida!")
            return True
        except Exception as e:
            logger.error(f"Falha no teste de conexão: {e}")
            return False

    def process_all_data(self):
        """Processa todos os dados e carrega no PostgreSQL"""
        try:
            # Validar arquivos
            self.validate_files()
            
            # Testar conexão antes de processar
            if not self.test_connection():
                raise ConnectionError("Não foi possível conectar ao banco de dados")
            
            # Conectar ao banco
            self.connect_database()
            
            # Criar estrutura do banco
            self.create_tables()
            
            # As tabelas já foram removidas e recriadas no create_tables()
            # Não precisa limpar dados pois as tabelas estão vazias
            
            # Carregar dados em ordem otimizada
            logger.info("Iniciando carregamento de dados...")
            
            # 1. Carregar entidades principais primeiro
            self.load_movies()
            self.load_links()
            self.load_genome_tags()
            
            # 2. Extrair e carregar usuários
            self.extract_users()
            
            # 3. Carregar dados relacionais
            self.load_ratings()
            self.load_user_tags()
            self.load_genome_scores()
            
            # 4. Analisar relacionamentos
            stats = self.analyze_relationships()
            
            logger.info("Processamento concluído com sucesso!")
            logger.info("Estatísticas finais:")
            for key, value in stats.items():
                logger.info(f"  {key}: {value}")
                
            return stats
            
        except Exception as e:
            logger.error(f"Erro durante o processamento: {e}")
            raise
        finally:
            if self.connection:
                self.connection.close()
                logger.info("Conexão com PostgreSQL fechada")
            if self.engine:
                self.engine.dispose()

def main():
    """Função principal"""
    print("="*60)
    print("PROCESSADOR DE DADOS DE FILMES - POSTGRESQL")
    print("="*60)
    print("Modo: Limpar e recarregar todos os dados")
    print("="*60)
    
    # Sempre limpar e recarregar
    processor = MovieDataProcessorPostgreSQL(clear_tables=True)
    stats = processor.process_all_data()
    
    print("\n" + "="*50)
    print("PROCESSAMENTO CONCLUÍDO - POSTGRESQL")
    print("="*50)
    print(f"Banco de dados: {processor.db_config['database']}")
    print(f"Host: {processor.db_config['host']}:{processor.db_config['port']}")
    print("Modo: Recarregar (limpar e recarregar)")
    print("\nEstatísticas:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    print("="*50)

if __name__ == "__main__":
    main()

