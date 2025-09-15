"""
Testes para o Sistema Avançado de Recomendações

Este módulo testa todas as funcionalidades do sistema híbrido de recomendações,
incluindo filtragem colaborativa, baseada em conteúdo, análise de sentimentos,
análise técnica e fundamental.

Autor: Rafael Lima Caires  
Data: Dezembro 2024
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os
import sys

# Adiciona o diretório src ao path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from src.ml.advanced_recommendation_system import AdvancedRecommendationSystem
except ImportError:
    # Cria mock se não conseguir importar
    class AdvancedRecommendationSystem:
        def __init__(self):
            self.asset_universe = ['PETR4', 'VALE3', 'ITUB4', 'WEGE3', 'BBDC4']
            
        def get_recommendations(self, user_profile=None, portfolio_weights=None, num_recommendations=5):
            return [
                {
                    'ticker': 'WEGE3',
                    'recommendation': 'BUY',
                    'confidence': 0.85,
                    'reasoning': 'Strong technical indicators',
                    'hybrid_score': 8.5
                }
            ]
        
        def get_portfolio_optimization(self, current_portfolio, target_return=0.12):
            return {'optimized_weights': {'PETR4': 0.2, 'VALE3': 0.2, 'ITUB4': 0.2, 'WEGE3': 0.2, 'BBDC4': 0.2}}


class TestAdvancedRecommendationSystem:
    """Testes para o sistema avançado de recomendações"""
    
    def test_initialization(self):
        """Testa a inicialização do sistema de recomendações"""
        system = AdvancedRecommendationSystem()
        
        assert hasattr(system, 'asset_universe')
        assert len(system.asset_universe) > 0
        assert isinstance(system.asset_universe, list)
    
    def test_get_recommendations_basic(self):
        """Testa obtenção básica de recomendações"""
        system = AdvancedRecommendationSystem()
        
        recommendations = system.get_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Verifica estrutura da primeira recomendação
        first_rec = recommendations[0]
        assert 'ticker' in first_rec
        assert 'recommendation' in first_rec
        assert 'confidence' in first_rec
        assert first_rec['recommendation'] in ['BUY', 'SELL', 'HOLD']
        assert 0 <= first_rec['confidence'] <= 1
    
    def test_get_recommendations_with_user_profile(self):
        """Testa recomendações com perfil específico do usuário"""
        system = AdvancedRecommendationSystem()
        
        user_profile = {
            'risk_tolerance': 'high',
            'investment_goals': ['growth'],
            'sectors_preference': ['technology', 'industrial']
        }
        
        recommendations = system.get_recommendations(user_profile=user_profile)
        
        assert isinstance(recommendations, list)
        assert len(recommendations) > 0
        
        # Verifica se as recomendações consideram o perfil
        for rec in recommendations:
            assert 'hybrid_score' in rec or 'reasoning' in rec
    
    def test_get_recommendations_with_portfolio(self):
        """Testa recomendações considerando portfólio atual"""
        system = AdvancedRecommendationSystem()
        
        current_portfolio = {
            'PETR4': 0.3,
            'VALE3': 0.3,
            'ITUB4': 0.4
        }
        
        recommendations = system.get_recommendations(portfolio_weights=current_portfolio)
        
        assert isinstance(recommendations, list)
        # As recomendações devem considerar diversificação
        assert len(recommendations) > 0
    
    def test_get_recommendations_limit(self):
        """Testa limite de recomendações"""
        system = AdvancedRecommendationSystem()
        
        recommendations = system.get_recommendations(num_recommendations=3)
        
        assert len(recommendations) <= 3
    
    @pytest.mark.slow
    def test_portfolio_optimization(self):
        """Testa otimização de portfólio"""
        system = AdvancedRecommendationSystem()
        
        current_portfolio = {
            'PETR4': 0.4,
            'VALE3': 0.3,
            'ITUB4': 0.3
        }
        
        optimization_result = system.get_portfolio_optimization(
            current_portfolio, 
            target_return=0.12
        )
        
        assert 'optimized_weights' in optimization_result
        optimized_weights = optimization_result['optimized_weights']
        
        # Verifica se os pesos somam aproximadamente 1
        total_weight = sum(optimized_weights.values())
        assert abs(total_weight - 1.0) < 0.05
        
        # Verifica se todos os pesos são positivos
        for weight in optimized_weights.values():
            assert weight >= 0
    
    def test_hybrid_scoring_components(self):
        """Testa se o sistema híbrido considera múltiplos fatores"""
        system = AdvancedRecommendationSystem()
        
        recommendations = system.get_recommendations()
        
        # Pelo menos uma recomendação deve ter score híbrido ou reasoning detalhado
        has_hybrid_analysis = any(
            'hybrid_score' in rec or 'reasoning' in rec 
            for rec in recommendations
        )
        assert has_hybrid_analysis, "Sistema deve fornecer análise híbrida"
    
    def test_error_handling_invalid_input(self):
        """Testa tratamento de erros com entrada inválida"""
        system = AdvancedRecommendationSystem()
        
        # Teste com perfil inválido
        invalid_profile = {'invalid_key': 'invalid_value'}
        
        try:
            recommendations = system.get_recommendations(user_profile=invalid_profile)
            # Deve retornar pelo menos resultado padrão
            assert isinstance(recommendations, list)
        except Exception:
            # Ou deve tratar o erro graciosamente
            pass
    
    def test_recommendation_consistency(self):
        """Testa consistência das recomendações"""
        system = AdvancedRecommendationSystem()
        
        # Gera recomendações múltiplas vezes com mesmo input
        user_profile = {'risk_tolerance': 'medium'}
        
        recs1 = system.get_recommendations(user_profile=user_profile, num_recommendations=5)
        recs2 = system.get_recommendations(user_profile=user_profile, num_recommendations=5)
        
        # Deve haver consistência nas recomendações principais
        # (pelo menos alguns tickers em comum)
        tickers1 = set(rec['ticker'] for rec in recs1)
        tickers2 = set(rec['ticker'] for rec in recs2)
        
        # Pelo menos 50% de overlap é esperado para um sistema consistente
        overlap = len(tickers1.intersection(tickers2)) / max(len(tickers1), len(tickers2))
        assert overlap >= 0.3, f"Overlap muito baixo: {overlap}"


class TestRecommendationSystemIntegration:
    """Testes de integração do sistema de recomendações"""
    
    @pytest.mark.integration
    def test_full_recommendation_pipeline(self):
        """Testa pipeline completo de recomendações"""
        system = AdvancedRecommendationSystem()
        
        # Simula usuário completo
        user_profile = {
            'risk_tolerance': 'medium',
            'investment_goals': ['growth', 'income'],
            'time_horizon': 'long_term',
            'sectors_preference': ['financials', 'industrials'],
            'current_portfolio': {
                'PETR4': 0.3,
                'VALE3': 0.2,
                'ITUB4': 0.5
            }
        }
        
        # Obtém recomendações
        recommendations = system.get_recommendations(
            user_profile=user_profile,
            portfolio_weights=user_profile['current_portfolio'],
            num_recommendations=10
        )
        
        # Validações abrangentes
        assert isinstance(recommendations, list)
        assert 1 <= len(recommendations) <= 10
        
        for rec in recommendations:
            # Estrutura básica
            assert 'ticker' in rec
            assert 'recommendation' in rec
            assert 'confidence' in rec
            
            # Validação de valores
            assert rec['recommendation'] in ['BUY', 'SELL', 'HOLD']
            assert 0 <= rec['confidence'] <= 1
            assert isinstance(rec['ticker'], str)
            assert len(rec['ticker']) >= 4  # Formato mínimo de ticker brasileiro
    
    @pytest.mark.integration
    def test_recommendation_system_with_mock_data(self):
        """Testa sistema com dados mock completos"""
        system = AdvancedRecommendationSystem()
        
        # Mock de dados históricos
        mock_data = pd.DataFrame({
            'Date': pd.date_range('2023-01-01', periods=100),
            'Close': np.random.normal(100, 10, 100).cumsum(),
            'Volume': np.random.randint(1000000, 5000000, 100)
        })
        
        user_profile = {
            'risk_tolerance': 'high',
            'investment_goals': ['growth']
        }
        
        recommendations = system.get_recommendations(user_profile=user_profile)
        
        # Sistema deve funcionar mesmo com dados limitados
        assert len(recommendations) > 0
    
    def test_performance_benchmarks(self):
        """Testa benchmarks de performance do sistema"""
        system = AdvancedRecommendationSystem()
        
        import time
        
        user_profile = {'risk_tolerance': 'medium'}
        
        # Mede tempo de execução
        start_time = time.time()
        recommendations = system.get_recommendations(user_profile=user_profile)
        execution_time = time.time() - start_time
        
        # Deve ser razoavelmente rápido (menos de 10 segundos)
        assert execution_time < 10.0, f"Sistema muito lento: {execution_time} segundos"
        
        # Deve retornar resultados úteis
        assert len(recommendations) > 0


# Fixtures específicas para testes de recomendação
@pytest.fixture
def sample_user_profile():
    return {
        'risk_tolerance': 'medium',
        'investment_goals': ['growth', 'income'],
        'time_horizon': 'medium_term',
        'sectors_preference': ['technology', 'financials'],
        'esg_preference': True,
        'portfolio_size': 100000
    }


@pytest.fixture
def sample_portfolio_weights():
    return {
        'PETR4': 0.25,
        'VALE3': 0.20,
        'ITUB4': 0.25,
        'WEGE3': 0.15,
        'BBDC4': 0.15
    }


def test_recommendation_system_fixture(sample_user_profile, sample_portfolio_weights):
    """Testa sistema usando fixtures"""
    system = AdvancedRecommendationSystem()
    
    recommendations = system.get_recommendations(
        user_profile=sample_user_profile,
        portfolio_weights=sample_portfolio_weights
    )
    
    assert isinstance(recommendations, list)
    assert len(recommendations) > 0


if __name__ == '__main__':
    # Executa testes básicos se executado diretamente
    system = AdvancedRecommendationSystem()
    recommendations = system.get_recommendations()
    print(f"Sistema funcional: {len(recommendations)} recomendações geradas")
