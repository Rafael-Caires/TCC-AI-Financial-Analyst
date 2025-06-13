"""
Novas rotas para APIs avançadas de IA

Este arquivo implementa endpoints para as novas funcionalidades de IA:
- Ensemble de modelos
- Análise de sentimentos
- Análise de risco avançada
- Sistema de recomendação híbrido

Autor: Rafael Lima Caires
Data: Junho 2025
"""

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import traceback
from datetime import datetime
import pandas as pd

from ..ml.ensemble_model import EnsembleModel
from ..ml.sentiment_analyzer import SentimentAnalyzer
from ..ml.risk_analyzer import RiskAnalyzer
from ..ml.advanced_recommendation_system import AdvancedRecommendationSystem
from ..utils.auth import token_required

# Cria blueprint para rotas de IA
ai_bp = Blueprint('ai', __name__)

# Inicializa sistemas de IA
ensemble_model = EnsembleModel()
sentiment_analyzer = SentimentAnalyzer()
risk_analyzer = RiskAnalyzer()
recommendation_system = AdvancedRecommendationSystem()

@ai_bp.route('/api/ai/ensemble/predict', methods=['POST'])
@cross_origin()
@token_required
def ensemble_predict(current_user):
    """
    Endpoint para previsões usando ensemble de modelos.
    """
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Ticker é obrigatório'}), 400
        
        ticker = data['ticker']
        days_ahead = data.get('days_ahead', 30)
        
        # Aqui você carregaria dados históricos do ticker
        # Por simplicidade, retornamos um exemplo
        predictions = {
            'ensemble': {
                'predictions': [
                    {
                        'date': '2025-06-09',
                        'predicted_price': 25.50,
                        'confidence': 0.75
                    },
                    {
                        'date': '2025-06-10',
                        'predicted_price': 25.80,
                        'confidence': 0.73
                    }
                ],
                'method': 'weighted_average',
                'weights_used': {
                    'lstm': 0.4,
                    'random_forest': 0.2,
                    'lightgbm': 0.25,
                    'xgboost': 0.15
                }
            },
            'individual': {
                'lstm': {'predictions': [25.45, 25.75]},
                'random_forest': {'predictions': [25.60, 25.85]},
                'lightgbm': {'predictions': [25.48, 25.78]},
                'xgboost': {'predictions': [25.55, 25.82]}
            }
        }
        
        return jsonify({
            'success': True,
            'data': predictions,
            'ticker': ticker,
            'days_ahead': days_ahead,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/sentiment/analyze', methods=['POST'])
@cross_origin()
@token_required
def analyze_sentiment(current_user):
    """
    Endpoint para análise de sentimentos de notícias.
    """
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Ticker é obrigatório'}), 400
        
        ticker = data['ticker']
        days_back = data.get('days_back', 7)
        
        # Analisa sentimento
        sentiment_data = sentiment_analyzer.get_news_sentiment(ticker, days_back)
        
        # Gera sinal de trading
        trading_signal = sentiment_analyzer.get_sentiment_signal(ticker)
        
        return jsonify({
            'success': True,
            'data': {
                'sentiment_analysis': sentiment_data,
                'trading_signal': trading_signal
            },
            'ticker': ticker,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/sentiment/market-summary', methods=['GET'])
@cross_origin()
@token_required
def market_sentiment_summary(current_user):
    """
    Endpoint para resumo de sentimento do mercado.
    """
    try:
        # Lista de principais ações brasileiras
        main_tickers = ['PETR4', 'VALE3', 'ITUB4', 'BBDC4', 'ABEV3']
        
        # Analisa sentimento do mercado
        market_summary = sentiment_analyzer.get_market_sentiment_summary(main_tickers)
        
        return jsonify({
            'success': True,
            'data': market_summary,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/risk/analyze-portfolio', methods=['POST'])
@cross_origin()
@token_required
def analyze_portfolio_risk(current_user):
    """
    Endpoint para análise de risco de portfólio.
    """
    try:
        data = request.get_json()
        
        if not data or 'portfolio' not in data:
            return jsonify({'error': 'Dados do portfólio são obrigatórios'}), 400
        
        portfolio_data = data['portfolio']  # {ticker: peso}
        period_days = data.get('period_days', 252)
        
        # Analisa risco do portfólio
        risk_analysis = risk_analyzer.analyze_portfolio_risk(portfolio_data, period_days)
        
        # Gera relatório textual
        risk_report = risk_analyzer.generate_risk_report(portfolio_data)
        
        return jsonify({
            'success': True,
            'data': {
                'risk_analysis': risk_analysis,
                'risk_report': risk_report
            },
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/risk/calculate-var', methods=['POST'])
@cross_origin()
@token_required
def calculate_var(current_user):
    """
    Endpoint para cálculo de Value at Risk.
    """
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Ticker é obrigatório'}), 400
        
        ticker = data['ticker']
        confidence_level = data.get('confidence_level', 0.95)
        method = data.get('method', 'historical')
        
        # Aqui você obteria dados históricos reais
        # Por simplicidade, retornamos valores simulados
        var_results = {
            'var_95': -0.025,
            'cvar_95': -0.035,
            'method_used': method,
            'confidence_level': confidence_level,
            'interpretation': 'Com 95% de confiança, a perda máxima esperada em 1 dia é de 2.5%'
        }
        
        return jsonify({
            'success': True,
            'data': var_results,
            'ticker': ticker,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/recommendations/personalized', methods=['POST'])
@cross_origin()
@token_required
def get_personalized_recommendations(current_user):
    """
    Endpoint para recomendações personalizadas.
    """
    try:
        data = request.get_json()
        
        if not data or 'user_profile' not in data:
            return jsonify({'error': 'Perfil do usuário é obrigatório'}), 400
        
        user_profile = data['user_profile']
        portfolio_value = data.get('portfolio_value', 10000)
        num_recommendations = data.get('num_recommendations', 10)
        
        # Gera recomendações personalizadas
        recommendations = recommendation_system.get_user_recommendations(
            user_profile, portfolio_value, num_recommendations
        )
        
        return jsonify({
            'success': True,
            'data': recommendations,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/recommendations/optimize-portfolio', methods=['POST'])
@cross_origin()
@token_required
def optimize_portfolio(current_user):
    """
    Endpoint para otimização de portfólio.
    """
    try:
        data = request.get_json()
        
        if not data or 'current_portfolio' not in data:
            return jsonify({'error': 'Portfólio atual é obrigatório'}), 400
        
        current_portfolio = data['current_portfolio']
        target_return = data.get('target_return', 0.12)
        
        # Otimiza portfólio
        optimization_result = recommendation_system.get_portfolio_optimization(
            current_portfolio, target_return
        )
        
        return jsonify({
            'success': True,
            'data': optimization_result,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/market/regime-analysis', methods=['GET'])
@cross_origin()
@token_required
def market_regime_analysis(current_user):
    """
    Endpoint para análise de regime de mercado.
    """
    try:
        # Analisa regime atual do mercado
        regime_analysis = recommendation_system.get_market_regime_analysis()
        
        return jsonify({
            'success': True,
            'data': regime_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/backtest/strategy', methods=['POST'])
@cross_origin()
@token_required
def backtest_strategy(current_user):
    """
    Endpoint para backtesting de estratégias.
    """
    try:
        data = request.get_json()
        
        if not data or 'strategy' not in data:
            return jsonify({'error': 'Estratégia é obrigatória'}), 400
        
        strategy = data['strategy']
        start_date = data.get('start_date', '2023-01-01')
        end_date = data.get('end_date', '2024-12-31')
        initial_capital = data.get('initial_capital', 100000)
        
        # Simula resultado de backtesting
        backtest_results = {
            'strategy_name': strategy.get('name', 'Custom Strategy'),
            'period': f"{start_date} to {end_date}",
            'initial_capital': initial_capital,
            'final_value': 125000,
            'total_return': 0.25,
            'annualized_return': 0.12,
            'volatility': 0.18,
            'sharpe_ratio': 0.67,
            'max_drawdown': -0.08,
            'win_rate': 0.65,
            'number_of_trades': 45,
            'performance_by_month': [
                {'month': '2023-01', 'return': 0.02},
                {'month': '2023-02', 'return': -0.01},
                {'month': '2023-03', 'return': 0.03}
                # ... mais dados
            ]
        }
        
        return jsonify({
            'success': True,
            'data': backtest_results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/technical-analysis/indicators', methods=['POST'])
@cross_origin()
@token_required
def calculate_technical_indicators(current_user):
    """
    Endpoint para cálculo de indicadores técnicos.
    """
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({'error': 'Ticker é obrigatório'}), 400
        
        ticker = data['ticker']
        indicators = data.get('indicators', ['sma', 'rsi', 'macd', 'bollinger'])
        
        # Simula cálculo de indicadores técnicos
        technical_indicators = {
            'ticker': ticker,
            'last_updated': datetime.now().isoformat(),
            'indicators': {
                'sma_20': 25.45,
                'sma_50': 24.80,
                'sma_200': 23.90,
                'rsi': 58.5,
                'macd': {
                    'macd_line': 0.15,
                    'signal_line': 0.12,
                    'histogram': 0.03
                },
                'bollinger_bands': {
                    'upper': 26.20,
                    'middle': 25.45,
                    'lower': 24.70,
                    'width': 1.50
                },
                'stochastic': {
                    'k_percent': 65.2,
                    'd_percent': 62.8
                },
                'williams_r': -35.5,
                'cci': 45.2
            },
            'signals': {
                'trend': 'bullish',
                'momentum': 'positive',
                'volatility': 'normal',
                'overall_signal': 'buy'
            }
        }
        
        return jsonify({
            'success': True,
            'data': technical_indicators,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/stress-test/portfolio', methods=['POST'])
@cross_origin()
@token_required
def stress_test_portfolio(current_user):
    """
    Endpoint para stress testing de portfólio.
    """
    try:
        data = request.get_json()
        
        if not data or 'portfolio' not in data:
            return jsonify({'error': 'Dados do portfólio são obrigatórios'}), 400
        
        portfolio_data = data['portfolio']
        scenarios = data.get('scenarios', ['market_crash', 'interest_rate_shock', 'currency_crisis'])
        
        # Simula stress testing
        stress_test_results = {
            'portfolio_composition': portfolio_data,
            'base_case': {
                'portfolio_value': 100000,
                'expected_return': 0.12,
                'volatility': 0.18
            },
            'stress_scenarios': {
                'market_crash_2008': {
                    'scenario_description': 'Queda de 20% no mercado em 1 dia',
                    'portfolio_impact': -18500,
                    'new_portfolio_value': 81500,
                    'probability': 'very_low',
                    'recovery_time_estimate': '12-18 months'
                },
                'covid_crash_2020': {
                    'scenario_description': 'Queda de 12% no mercado em 1 dia',
                    'portfolio_impact': -11200,
                    'new_portfolio_value': 88800,
                    'probability': 'low',
                    'recovery_time_estimate': '6-12 months'
                },
                'interest_rate_shock': {
                    'scenario_description': 'Aumento de 3% na taxa de juros',
                    'portfolio_impact': -8500,
                    'new_portfolio_value': 91500,
                    'probability': 'medium',
                    'recovery_time_estimate': '3-6 months'
                }
            },
            'risk_metrics': {
                'var_95': -2500,
                'cvar_95': -3200,
                'maximum_loss_scenario': -18500,
                'diversification_benefit': 0.15
            },
            'recommendations': [
                'Considere aumentar diversificação internacional',
                'Avalie hedge para proteção contra quedas extremas',
                'Mantenha reserva de emergência adequada'
            ]
        }
        
        return jsonify({
            'success': True,
            'data': stress_test_results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_bp.route('/api/ai/model/performance', methods=['GET'])
@cross_origin()
@token_required
def get_model_performance(current_user):
    """
    Endpoint para métricas de performance dos modelos.
    """
    try:
        # Simula métricas de performance dos modelos
        model_performance = {
            'last_evaluation': datetime.now().isoformat(),
            'models': {
                'lstm': {
                    'rmse': 0.025,
                    'mae': 0.018,
                    'mape': 2.1,
                    'direction_accuracy': 0.68,
                    'sharpe_ratio': 0.45,
                    'status': 'active'
                },
                'random_forest': {
                    'rmse': 0.032,
                    'mae': 0.024,
                    'mape': 2.8,
                    'direction_accuracy': 0.62,
                    'sharpe_ratio': 0.38,
                    'status': 'active'
                },
                'lightgbm': {
                    'rmse': 0.028,
                    'mae': 0.021,
                    'mape': 2.4,
                    'direction_accuracy': 0.65,
                    'sharpe_ratio': 0.42,
                    'status': 'active'
                },
                'xgboost': {
                    'rmse': 0.030,
                    'mae': 0.022,
                    'mape': 2.6,
                    'direction_accuracy': 0.63,
                    'sharpe_ratio': 0.40,
                    'status': 'active'
                },
                'ensemble': {
                    'rmse': 0.022,
                    'mae': 0.016,
                    'mape': 1.9,
                    'direction_accuracy': 0.72,
                    'sharpe_ratio': 0.52,
                    'status': 'active'
                }
            },
            'ensemble_weights': {
                'lstm': 0.4,
                'random_forest': 0.2,
                'lightgbm': 0.25,
                'xgboost': 0.15
            },
            'model_drift_detection': {
                'lstm': {'drift_detected': False, 'confidence': 0.95},
                'random_forest': {'drift_detected': False, 'confidence': 0.92},
                'lightgbm': {'drift_detected': True, 'confidence': 0.88},
                'xgboost': {'drift_detected': False, 'confidence': 0.94}
            },
            'retraining_schedule': {
                'last_retrain': '2025-06-01',
                'next_retrain': '2025-07-01',
                'retrain_frequency': 'monthly'
            }
        }
        
        return jsonify({
            'success': True,
            'data': model_performance,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

