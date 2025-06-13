"""
Configurações da aplicação Flask

Este arquivo contém as configurações da aplicação, separadas por ambiente
(desenvolvimento, teste, produção).

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import os
from dotenv import load_dotenv

# Carrega variáveis de ambiente
load_dotenv()

class Config:
    """Configuração base"""
    SECRET_KEY = os.environ.get('SECRET_KEY', 'dev_key_change_in_production')
    DEBUG = False
    TESTING = False
    
    # Configurações do banco de dados (descomente se necessário)
    # SQLALCHEMY_DATABASE_URI = f"mysql+pymysql://{os.getenv('DB_USERNAME', 'root')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'finance_db')}"
    # SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # Configurações de cache
    CACHE_TYPE = 'simple'
    
    # Configurações de API externa (Yahoo Finance, Alpha Vantage, etc.)
    YAHOO_FINANCE_API_KEY = os.environ.get('YAHOO_FINANCE_API_KEY', '')
    ALPHA_VANTAGE_API_KEY = os.environ.get('ALPHA_VANTAGE_API_KEY', '')
    
    # Configurações de modelos de ML
    MODEL_SAVE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')


class DevelopmentConfig(Config):
    """Configuração de desenvolvimento"""
    DEBUG = True
    

class TestingConfig(Config):
    """Configuração de teste"""
    TESTING = True
    DEBUG = True
    

class ProductionConfig(Config):
    """Configuração de produção"""
    # Em produção, todas as chaves secretas devem vir de variáveis de ambiente
    SECRET_KEY = os.environ.get('SECRET_KEY')
    

# Dicionário de configurações por ambiente
config_by_name = {
    'development': DevelopmentConfig,
    'testing': TestingConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

# Obtém a configuração atual com base na variável de ambiente
def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    return config_by_name.get(env, config_by_name['default'])
