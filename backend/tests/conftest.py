"""
Configuração de testes para o sistema financeiro com IA

Este arquivo contém a configuração base para todos os testes do sistema,
incluindo fixtures, mocks e utilitários comuns.

Autor: Rafael Lima Caires
Data: Dezembro 2024
"""

import pytest
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock
import tempfile
import json

# Adiciona o diretório src ao path para importações
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Importações do sistema
from src.ml.advanced_recommendation_system import AdvancedRecommendationSystem
from src.ml.risk_analyzer import AdvancedRiskAnalyzer
from src.ml.model_integrator import AdvancedModelIntegrator
from src.ml.sentiment_analyzer import SentimentAnalyzer


class TestConfig:
    """Configuração para ambiente de testes"""
    TESTING = True
    SECRET_KEY = 'test-secret-key'
    DATABASE_URL = 'sqlite:///:memory:'


@pytest.fixture
def mock_stock_data():
    """
    Fixture que fornece dados simulados de ações para testes.
    """
    np.random.seed(42)  # Para resultados reproduzíveis
    
    dates = pd.date_range(start='2023-01-01', end='2024-12-31', freq='D')
    prices = []
    
    # Gera série temporal com tendência e ruído
    base_price = 100.0
    trend = 0.0002  # Tendência ligeiramente positiva
    volatility = 0.02
    
    for i, date in enumerate(dates):
        # Aplica tendência e volatilidade
        return_daily = trend + np.random.normal(0, volatility)
        new_price = base_price * (1 + return_daily)
        
        # Simula dados OHLCV
        high = new_price * (1 + abs(np.random.normal(0, 0.01)))
        low = new_price * (1 - abs(np.random.normal(0, 0.01)))
        volume = int(np.random.normal(1000000, 200000))
        
        prices.append({
            'Date': date,
            'Open': base_price if i == 0 else prices[i-1]['Close'],
            'High': high,
            'Low': low,
            'Close': new_price,
            'Volume': max(volume, 100000)
        })
        
        base_price = new_price
    
    df = pd.DataFrame(prices)
    df.set_index('Date', inplace=True)
    return df


@pytest.fixture
def sample_portfolio():
    """
    Fixture que fornece dados de portfólio para testes.
    """
    return {
        'PETR4': 0.25,
        'VALE3': 0.20,
        'ITUB4': 0.20,
        'WEGE3': 0.15,
        'BBDC4': 0.20
    }


@pytest.fixture
def mock_user_profile():
    """
    Fixture que fornece perfil de usuário para testes.
    """
    return {
        'risk_tolerance': 'medium',
        'investment_goals': ['growth', 'income'],
        'time_horizon': 'long_term',
        'sectors_preference': ['technology', 'financials'],
        'esg_preference': True
    }


@pytest.fixture
def temp_models_dir():
    """
    Fixture que cria um diretório temporário para modelos de ML.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.fixture
def mock_news_data():
    """
    Fixture que fornece dados simulados de notícias para testes de sentiment.
    """
    return [
        {
            'title': 'Petrobras anuncia aumento de dividendos',
            'content': 'A Petrobras anunciou um aumento significativo nos dividendos para os acionistas.',
            'date': datetime.now() - timedelta(days=1),
            'sentiment': 'positive',
            'source': 'valor_economico'
        },
        {
            'title': 'Incertezas no mercado afetam ações',
            'content': 'As incertezas econômicas globais afetam o desempenho das ações brasileiras.',
            'date': datetime.now() - timedelta(days=2),
            'sentiment': 'negative',
            'source': 'folha'
        },
        {
            'title': 'Resultados mistos no setor financeiro',
            'content': 'Os bancos apresentaram resultados mistos no último trimestre.',
            'date': datetime.now() - timedelta(days=3),
            'sentiment': 'neutral',
            'source': 'estadao'
        }
    ]


@pytest.fixture
def recommendation_system():
    """
    Fixture que fornece uma instância do sistema de recomendações.
    """
    return AdvancedRecommendationSystem()


@pytest.fixture
def risk_analyzer():
    """
    Fixture que fornece uma instância do analisador de risco.
    """
    return AdvancedRiskAnalyzer()


@pytest.fixture
def model_integrator(temp_models_dir):
    """
    Fixture que fornece uma instância do integrador de modelos.
    """
    return AdvancedModelIntegrator(models_dir=temp_models_dir)


@pytest.fixture
def sentiment_analyzer():
    """
    Fixture que fornece uma instância do analisador de sentimentos.
    """
    return SentimentAnalyzer()


def create_mock_flask_app():
    """
    Cria uma aplicação Flask mock para testes.
    """
    from flask import Flask
    
    app = Flask(__name__)
    app.config.from_object(TestConfig)
    
    return app


def assert_portfolio_weights(weights_dict):
    """
    Utilitário para validar se os pesos do portfólio somam 1.0.
    """
    total_weight = sum(weights_dict.values())
    assert abs(total_weight - 1.0) < 0.01, f"Pesos do portfólio somam {total_weight}, esperado ~1.0"


def assert_valid_recommendation(recommendation):
    """
    Utilitário para validar estrutura de recomendação.
    """
    required_fields = ['ticker', 'recommendation', 'confidence', 'reasoning']
    for field in required_fields:
        assert field in recommendation, f"Campo obrigatório '{field}' não encontrado na recomendação"
    
    assert isinstance(recommendation['confidence'], (int, float)), "Confiança deve ser numérica"
    assert 0 <= recommendation['confidence'] <= 1, "Confiança deve estar entre 0 e 1"
    assert recommendation['recommendation'] in ['BUY', 'SELL', 'HOLD'], "Recomendação deve ser BUY, SELL ou HOLD"


def assert_valid_risk_metrics(risk_metrics):
    """
    Utilitário para validar métricas de risco.
    """
    required_metrics = ['var_95', 'cvar_95', 'volatility', 'sharpe_ratio', 'max_drawdown']
    for metric in required_metrics:
        assert metric in risk_metrics, f"Métrica de risco '{metric}' não encontrada"
        assert isinstance(risk_metrics[metric], (int, float, np.float64)), f"Métrica '{metric}' deve ser numérica"


class MockDataProvider:
    """
    Classe mock para simulação de provedor de dados externos.
    """
    
    @staticmethod
    def get_stock_price(ticker):
        """Simula obtenção de preço de ação"""
        prices = {
            'PETR4': 25.50,
            'VALE3': 65.80,
            'ITUB4': 32.45,
            'BBDC4': 28.90,
            'WEGE3': 45.20
        }
        return prices.get(ticker, 100.0)
    
    @staticmethod
    def get_historical_data(ticker, start_date, end_date):
        """Simula obtenção de dados históricos"""
        # Retorna dados simulados baseados na fixture mock_stock_data
        np.random.seed(hash(ticker) % 1000)
        dates = pd.date_range(start=start_date, end=end_date, freq='D')
        
        data = []
        base_price = MockDataProvider.get_stock_price(ticker)
        
        for date in dates:
            return_daily = np.random.normal(0.001, 0.02)
            price = base_price * (1 + return_daily)
            
            data.append({
                'Date': date,
                'Open': base_price,
                'High': price * 1.02,
                'Low': price * 0.98,
                'Close': price,
                'Volume': int(np.random.normal(1000000, 200000))
            })
            
            base_price = price
        
        df = pd.DataFrame(data)
        df.set_index('Date', inplace=True)
        return df
    
    @staticmethod
    def get_market_data():
        """Simula dados de mercado"""
        return {
            'ibovespa': {
                'level': 118500.25,
                'change': 1.23,
                'volume': 15000000000
            },
            'dollar': {
                'rate': 5.25,
                'change': -0.45
            },
            'selic': {
                'rate': 11.75,
                'last_meeting': '2024-12-11'
            }
        }


@pytest.fixture
def mock_data_provider():
    """
    Fixture que fornece o provider de dados mock.
    """
    return MockDataProvider


# Decorador para marcar testes que requerem dados externos
slow_test = pytest.mark.slow


# Decorador para marcar testes de integração
integration_test = pytest.mark.integration


# Decorador para marcar testes que requerem ML models
ml_test = pytest.mark.ml


# Configurações para diferentes tipos de teste
pytest_plugins = []
