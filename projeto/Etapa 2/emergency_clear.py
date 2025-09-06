#!/usr/bin/env python3
"""
Script de emergência para limpar as tabelas do PostgreSQL
Use este script se o processo principal estiver travado
"""

import psycopg2
import configparser
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_database_config():
    """Carrega configurações do banco de dados"""
    config = configparser.ConfigParser()
    
    # Configurações padrão
    default_config = {
        'host': 'localhost',
        'port': '5432',
        'database': 'movie_database',
        'user': 'postgres',
        'password': '123'
    }
    
    try:
        config.read('database_config.ini', encoding='utf-8')
        if 'postgresql' in config:
            return dict(config['postgresql'])
    except:
        pass
    
    return default_config

def emergency_clear():
    """Limpa as tabelas de forma rápida"""
    db_config = load_database_config()
    
    try:
        # Conectar ao banco
        connection = psycopg2.connect(
            host=db_config['host'],
            port=db_config['port'],
            database=db_config['database'],
            user=db_config['user'],
            password=db_config['password']
        )
        
        cursor = connection.cursor()
        
        # Desabilitar autocommit
        connection.autocommit = False
        
        # Tabelas para limpar
        tables = [
            'genome_scores',
            'user_tags', 
            'ratings',
            'users',
            'genome_tags',
            'movie_links',
            'movies'
        ]
        
        logger.info("Iniciando limpeza de emergência...")
        
        for table in tables:
            try:
                logger.info(f"Truncando tabela {table}...")
                cursor.execute(f"TRUNCATE TABLE {table} RESTART IDENTITY CASCADE")
                logger.info(f"Tabela {table} truncada com sucesso")
            except Exception as e:
                logger.warning(f"Erro ao truncar {table}: {e}")
                try:
                    cursor.execute(f"DELETE FROM {table}")
                    logger.info(f"Tabela {table} limpa com DELETE")
                except Exception as e2:
                    logger.error(f"Erro ao deletar de {table}: {e2}")
        
        connection.commit()
        logger.info("Limpeza de emergência concluída!")
        
    except Exception as e:
        logger.error(f"Erro na limpeza de emergência: {e}")
    finally:
        if 'connection' in locals():
            connection.close()

if __name__ == "__main__":
    emergency_clear()
