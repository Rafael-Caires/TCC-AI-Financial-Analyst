"""
Rotas para Portfolio Avançado

Este arquivo implementa endpoints para funcionalidades avançadas do portfólio:
- Portfolio detalhado com métricas de risco
- Análise de performance histórica
- Comparação com benchmarks
- Otimização e rebalanceamento
- Stress testing

Autor: Rafael Lima Caires
Data: Dezembro 2024
"""

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import traceback
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from ..ml.risk_analyzer import RiskAnalyzer
from ..ml.advanced_recommendation_system import AdvancedRecommendationSystem
from ..utils.auth import token_required

# Cria blueprint para rotas de portfólio
portfolio_bp = Blueprint('portfolio_advanced', __name__)

# Inicializa componentes de análise
risk_analyzer = RiskAnalyzer()
recommendation_system = AdvancedRecommendationSystem()

@portfolio_bp.route('/api/portfolio/detailed', methods=['GET'])
@cross_origin()
def get_detailed_portfolio():
    """
    Endpoint para obter portfólio detalhado com métricas avançadas.
    """
    try:
        user_id = 1  # Usuário simulado para desenvolvimento
        
        # Simula dados detalhados do portfólio
        portfolio_data = {
            'summary': {
                'total_value': 145230.50,
                'total_invested': 120000.00,
                'total_return': 25230.50,
                'return_percentage': 21.03,
                'daily_change': 1250.30,
                'daily_change_percentage': 0.87,
                'last_updated': datetime.now().isoformat()
            },
            'assets': [
                {
                    'ticker': 'PETR4',
                    'name': 'Petrobras PN',
                    'sector': 'Energia',
                    'quantity': 500,
                    'avg_price': 28.50,
                    'current_price': 32.45,
                    'total_invested': 14250.00,
                    'current_value': 16225.00,
                    'return_value': 1975.00,
                    'return_percentage': 13.86,
                    'allocation_percentage': 11.2,
                    'daily_change': 0.85,
                    'daily_change_percentage': 2.69,
                    'risk_score': 0.75,
                    'beta': 1.25,
                    'volatility': 0.28,
                    'dividend_yield': 0.042,
                    'pe_ratio': 15.2,
                    'recommendation': 'HOLD',
                    'last_price_update': datetime.now().isoformat()
                },
                {
                    'ticker': 'VALE3',
                    'name': 'Vale ON',
                    'sector': 'Mineração',
                    'quantity': 200,
                    'avg_price': 65.00,
                    'current_price': 71.20,
                    'total_invested': 13000.00,
                    'current_value': 14240.00,
                    'return_value': 1240.00,
                    'return_percentage': 9.54,
                    'allocation_percentage': 9.8,
                    'daily_change': 1.45,
                    'daily_change_percentage': 2.08,
                    'risk_score': 0.78,
                    'beta': 1.18,
                    'volatility': 0.32,
                    'dividend_yield': 0.055,
                    'pe_ratio': 12.8,
                    'recommendation': 'BUY',
                    'last_price_update': datetime.now().isoformat()
                },
                {
                    'ticker': 'ITUB4',
                    'name': 'Itaú Unibanco PN',
                    'sector': 'Financeiro',
                    'quantity': 800,
                    'avg_price': 24.00,
                    'current_price': 27.30,
                    'total_invested': 19200.00,
                    'current_value': 21840.00,
                    'return_value': 2640.00,
                    'return_percentage': 13.75,
                    'allocation_percentage': 15.0,
                    'daily_change': 0.45,
                    'daily_change_percentage': 1.68,
                    'risk_score': 0.65,
                    'beta': 1.05,
                    'volatility': 0.22,
                    'dividend_yield': 0.035,
                    'pe_ratio': 11.5,
                    'recommendation': 'HOLD',
                    'last_price_update': datetime.now().isoformat()
                },
                {
                    'ticker': 'WEGE3',
                    'name': 'WEG ON',
                    'sector': 'Industrial',
                    'quantity': 300,
                    'avg_price': 42.00,
                    'current_price': 48.50,
                    'total_invested': 12600.00,
                    'current_value': 14550.00,
                    'return_value': 1950.00,
                    'return_percentage': 15.48,
                    'allocation_percentage': 10.0,
                    'daily_change': 0.85,
                    'daily_change_percentage': 1.79,
                    'risk_score': 0.55,
                    'beta': 0.95,
                    'volatility': 0.18,
                    'dividend_yield': 0.018,
                    'pe_ratio': 18.3,
                    'recommendation': 'BUY',
                    'last_price_update': datetime.now().isoformat()
                },
                {
                    'ticker': 'BBDC4',
                    'name': 'Bradesco PN',
                    'sector': 'Financeiro',
                    'quantity': 1000,
                    'avg_price': 15.50,
                    'current_price': 16.25,
                    'total_invested': 15500.00,
                    'current_value': 16250.00,
                    'return_value': 750.00,
                    'return_percentage': 4.84,
                    'allocation_percentage': 11.2,
                    'daily_change': 0.15,
                    'daily_change_percentage': 0.93,
                    'risk_score': 0.68,
                    'beta': 1.08,
                    'volatility': 0.24,
                    'dividend_yield': 0.038,
                    'pe_ratio': 10.8,
                    'recommendation': 'HOLD',
                    'last_price_update': datetime.now().isoformat()
                }
            ],
            'allocation_by_sector': [
                {
                    'sector': 'Financeiro',
                    'percentage': 26.2,
                    'value': 38090.00,
                    'count': 2,
                    'avg_return': 9.30,
                    'risk_score': 0.665
                },
                {
                    'sector': 'Energia',
                    'percentage': 11.2,
                    'value': 16225.00,
                    'count': 1,
                    'avg_return': 13.86,
                    'risk_score': 0.75
                },
                {
                    'sector': 'Mineração',
                    'percentage': 9.8,
                    'value': 14240.00,
                    'count': 1,
                    'avg_return': 9.54,
                    'risk_score': 0.78
                },
                {
                    'sector': 'Industrial',
                    'percentage': 10.0,
                    'value': 14550.00,
                    'count': 1,
                    'avg_return': 15.48,
                    'risk_score': 0.55
                },
                {
                    'sector': 'Outros',
                    'percentage': 42.8,
                    'value': 62125.50,
                    'count': 8,
                    'avg_return': 12.5,
                    'risk_score': 0.62
                }
            ],
            'metrics': {
                'beta': 1.15,
                'alpha': 0.025,
                'sharpe_ratio': 0.68,
                'sortino_ratio': 0.85,
                'volatility': 0.24,
                'var_95': -0.048,
                'cvar_95': -0.065,
                'max_drawdown': -0.18,
                'calmar_ratio': 1.17,
                'correlation_ibov': 0.82,
                'tracking_error': 0.045,
                'information_ratio': 0.33,
                'treynor_ratio': 0.142
            },
            'performance_metrics': {
                '1D': {'return': 0.87, 'vs_benchmark': 0.23},
                '1W': {'return': 3.2, 'vs_benchmark': 1.1},
                '1M': {'return': 5.8, 'vs_benchmark': 2.3},
                '3M': {'return': 12.1, 'vs_benchmark': 4.2},
                '6M': {'return': 18.7, 'vs_benchmark': 7.1},
                '1Y': {'return': 21.03, 'vs_benchmark': 8.5},
                'YTD': {'return': 19.2, 'vs_benchmark': 7.8}
            }
        }
        
        return jsonify({
            'success': True,
            'data': portfolio_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@portfolio_bp.route('/api/portfolio/performance', methods=['GET'])
@cross_origin()
def get_performance_history():
    """
    Endpoint para obter histórico de performance do portfólio.
    """
    try:
        timeframe = request.args.get('timeframe', '1Y')
        
        # Gera dados históricos baseados no timeframe
        if timeframe == '1M':
            periods = 30
            labels = [(datetime.now() - timedelta(days=i)).strftime('%d/%m') for i in reversed(range(periods))]
        elif timeframe == '3M':
            periods = 90
            labels = [(datetime.now() - timedelta(days=i)).strftime('%d/%m') for i in reversed(range(0, periods, 3))]
        elif timeframe == '1Y':
            periods = 12
            labels = [(datetime.now() - timedelta(days=30*i)).strftime('%m/%Y') for i in reversed(range(periods))]
        else:
            periods = 12
            labels = [f'Mês {i+1}' for i in range(periods)]
        
        # Simula retornos mensais
        np.random.seed(42)
        portfolio_returns = np.random.normal(1.5, 2.5, periods).round(1)
        benchmark_returns = np.random.normal(1.2, 2.0, periods).round(1)
        
        monthly_returns = [
            {
                'period': labels[i],
                'portfolio': float(portfolio_returns[i]),
                'benchmark': float(benchmark_returns[i]),
                'excess_return': float(portfolio_returns[i] - benchmark_returns[i])
            }
            for i in range(periods)
        ]
        
        # Calcula performance cumulativa
        portfolio_cumulative = np.cumprod(1 + portfolio_returns/100)
        benchmark_cumulative = np.cumprod(1 + benchmark_returns/100)
        
        cumulative_performance = [
            {
                'period': labels[i],
                'portfolio': float((portfolio_cumulative[i] - 1) * 100),
                'benchmark': float((benchmark_cumulative[i] - 1) * 100)
            }
            for i in range(periods)
        ]
        
        performance_data = {
            'timeframe': timeframe,
            'monthly_returns': monthly_returns,
            'cumulative_performance': cumulative_performance,
            'summary': {
                'total_return': float((portfolio_cumulative[-1] - 1) * 100),
                'benchmark_return': float((benchmark_cumulative[-1] - 1) * 100),
                'excess_return': float((portfolio_cumulative[-1] - benchmark_cumulative[-1]) * 100),
                'volatility': float(np.std(portfolio_returns)),
                'sharpe_ratio': float(np.mean(portfolio_returns) / np.std(portfolio_returns)),
                'best_month': float(np.max(portfolio_returns)),
                'worst_month': float(np.min(portfolio_returns)),
                'positive_months': int(np.sum(portfolio_returns > 0)),
                'negative_months': int(np.sum(portfolio_returns < 0))
            }
        }
        
        return jsonify({
            'success': True,
            'data': performance_data,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@portfolio_bp.route('/api/portfolio/benchmark', methods=['GET'])
@cross_origin()
def get_benchmark_comparison():
    """
    Endpoint para comparação com benchmarks.
    """
    try:
        benchmark = request.args.get('benchmark', 'IBOV')
        
        # Dados de comparação com diferentes benchmarks
        benchmark_data = {
            'IBOV': {
                'name': 'Ibovespa',
                'current_level': 118500.25,
                'daily_change': 0.45,
                'ytd_return': 14.3,
                'correlation': 0.82,
                'beta': 1.15
            },
            'IBRX': {
                'name': 'IBrX-100',
                'current_level': 22345.80,
                'daily_change': 0.38,
                'ytd_return': 13.8,
                'correlation': 0.85,
                'beta': 1.08
            },
            'SMLL': {
                'name': 'Small Caps',
                'current_level': 3245.60,
                'daily_change': 0.62,
                'ytd_return': 18.2,
                'correlation': 0.65,
                'beta': 1.35
            },
            'CDI': {
                'name': 'CDI',
                'current_level': 100.0,
                'daily_change': 0.04,
                'ytd_return': 10.5,
                'correlation': -0.12,
                'beta': 0.05
            }
        }
        
        selected_benchmark = benchmark_data.get(benchmark, benchmark_data['IBOV'])
        
        # Análise comparativa detalhada
        comparison_data = {
            'benchmark_info': selected_benchmark,
            'portfolio_vs_benchmark': {
                'excess_return_ytd': 19.2 - selected_benchmark['ytd_return'],
                'tracking_error': 4.5,
                'information_ratio': 1.12,
                'up_capture': 105.2,
                'down_capture': 92.8,
                'correlation': selected_benchmark['correlation'],
                'beta': selected_benchmark['beta']
            },
            'risk_adjusted_metrics': {
                'portfolio_sharpe': 0.68,
                'benchmark_sharpe': 0.52,
                'portfolio_sortino': 0.85,
                'benchmark_sortino': 0.68,
                'portfolio_calmar': 1.17,
                'benchmark_calmar': 0.94
            },
            'attribution_analysis': {
                'asset_selection': 3.2,
                'sector_allocation': 1.8,
                'interaction_effect': -0.5,
                'total_active_return': 4.5
            },
            'historical_comparison': [
                {'period': '1M', 'portfolio': 5.8, 'benchmark': 3.5, 'excess': 2.3},
                {'period': '3M', 'portfolio': 12.1, 'benchmark': 7.9, 'excess': 4.2},
                {'period': '6M', 'portfolio': 18.7, 'benchmark': 11.6, 'excess': 7.1},
                {'period': '1Y', 'portfolio': 21.0, 'benchmark': 12.5, 'excess': 8.5}
            ]
        }
        
        return jsonify({
            'success': True,
            'data': comparison_data,
            'benchmark': benchmark,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@portfolio_bp.route('/api/risk-analysis/portfolio', methods=['POST'])
@cross_origin()
def analyze_portfolio_risk():
    """
    Endpoint para análise completa de risco do portfólio.
    """
    try:
        data = request.get_json()
        
        if not data or 'portfolio_weights' not in data:
            return jsonify({'error': 'Pesos do portfólio são obrigatórios'}), 400
        
        portfolio_weights = data['portfolio_weights']
        confidence_level = data.get('confidence_level', 0.95)
        time_horizon = data.get('time_horizon', 252)  # dias úteis
        
        # Análise avançada de risco
        risk_analysis = {
            'risk_metrics': {
                'var_95': -0.048,
                'var_99': -0.072,
                'cvar_95': -0.065,
                'cvar_99': -0.095,
                'volatility': 0.24,
                'annualized_volatility': 0.24 * np.sqrt(252),
                'beta': 1.15,
                'alpha': 0.025,
                'sharpe_ratio': 0.68,
                'sortino_ratio': 0.85,
                'max_drawdown': -0.18,
                'calmar_ratio': 1.17,
                'omega_ratio': 1.24
            },
            'diversification': {
                'herfindahl_index': 0.15,
                'effective_number_of_stocks': 6.67,
                'sector_concentration': 0.262,
                'geographic_concentration': 1.0,  # Brasil = 100%
                'diversification_ratio': 0.78,
                'concentration_risk': 'Medium'
            },
            'correlation_analysis': {
                'average_correlation': 0.35,
                'max_correlation': 0.68,
                'min_correlation': -0.12,
                'correlation_with_market': 0.82,
                'correlation_clusters': [
                    {'assets': ['ITUB4', 'BBDC4'], 'avg_correlation': 0.68},
                    {'assets': ['PETR4', 'VALE3'], 'avg_correlation': 0.45}
                ]
            },
            'stress_testing': {
                'market_crash_scenario': {
                    'scenario': 'Queda de 20% no Ibovespa',
                    'portfolio_impact': -18.5,
                    'probability': 0.05,
                    'recovery_estimate': '12-18 meses'
                },
                'interest_rate_shock': {
                    'scenario': 'Aumento de 3pp na Selic',
                    'portfolio_impact': -8.2,
                    'probability': 0.15,
                    'recovery_estimate': '6-12 meses'
                },
                'currency_crisis': {
                    'scenario': 'Desvalorização de 25% do Real',
                    'portfolio_impact': -12.8,
                    'probability': 0.10,
                    'recovery_estimate': '9-15 meses'
                }
            },
            'regime_analysis': {
                'current_regime': 'Normal',
                'regime_probability': {
                    'bull_market': 0.35,
                    'normal_market': 0.50,
                    'bear_market': 0.15
                },
                'regime_specific_metrics': {
                    'bull_market': {'expected_return': 0.18, 'volatility': 0.16},
                    'normal_market': {'expected_return': 0.12, 'volatility': 0.24},
                    'bear_market': {'expected_return': -0.08, 'volatility': 0.35}
                }
            },
            'liquidity_analysis': {
                'portfolio_liquidity_score': 0.75,
                'days_to_liquidate': {
                    '25%': 2,
                    '50%': 5,
                    '75%': 12,
                    '100%': 25
                },
                'bid_ask_impact': 0.012
            },
            'risk_contribution': {
                'PETR4': {'risk_contribution': 0.18, 'allocation': 0.112},
                'VALE3': {'risk_contribution': 0.15, 'allocation': 0.098},
                'ITUB4': {'risk_contribution': 0.22, 'allocation': 0.150},
                'WEGE3': {'risk_contribution': 0.12, 'allocation': 0.100},
                'BBDC4': {'risk_contribution': 0.16, 'allocation': 0.112}
            }
        }
        
        return jsonify({
            'success': True,
            'data': risk_analysis,
            'confidence_level': confidence_level,
            'time_horizon': time_horizon,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@portfolio_bp.route('/api/portfolio/optimize', methods=['POST'])
@cross_origin()
def optimize_portfolio():
    """
    Endpoint para otimização de portfólio.
    """
    try:
        data = request.get_json()
        
        current_portfolio = data.get('current_portfolio', {})
        target_return = data.get('target_return', 0.12)
        risk_tolerance = data.get('risk_tolerance', 'medium')
        constraints = data.get('constraints', {})
        
        # Simulação de otimização de portfólio
        optimization_results = {
            'optimization_method': 'Mean-Variance Optimization',
            'objective': 'Maximize Sharpe Ratio',
            'constraints_applied': constraints,
            'current_portfolio': current_portfolio,
            'optimized_portfolio': {
                'PETR4': 0.08,  # Reduzir exposição
                'VALE3': 0.10,
                'ITUB4': 0.18,  # Aumentar peso
                'WEGE3': 0.15,  # Aumentar peso
                'BBDC4': 0.08,  # Reduzir exposição
                'ABEV3': 0.12,  # Nova posição
                'MGLU3': 0.08,  # Nova posição
                'RENT3': 0.10,  # Nova posição
                'GGBR4': 0.06,  # Nova posição
                'SUZB3': 0.05   # Nova posição
            },
            'expected_metrics': {
                'expected_return': 0.145,
                'expected_volatility': 0.22,
                'expected_sharpe_ratio': 0.76,
                'var_95': -0.042,
                'max_drawdown_estimate': -0.15
            },
            'rebalancing_needed': {
                'total_trades': 7,
                'estimated_cost': 450.00,
                'net_expected_benefit': 2800.00,
                'trades': [
                    {'action': 'SELL', 'ticker': 'PETR4', 'current_weight': 0.112, 'target_weight': 0.08, 'amount': -4650.00},
                    {'action': 'BUY', 'ticker': 'WEGE3', 'current_weight': 0.10, 'target_weight': 0.15, 'amount': 7260.00},
                    {'action': 'BUY', 'ticker': 'ABEV3', 'current_weight': 0.0, 'target_weight': 0.12, 'amount': 17427.60},
                    {'action': 'BUY', 'ticker': 'RENT3', 'current_weight': 0.0, 'target_weight': 0.10, 'amount': 14523.00}
                ]
            },
            'alternative_strategies': [
                {
                    'name': 'Risk Parity',
                    'expected_return': 0.132,
                    'expected_volatility': 0.19,
                    'sharpe_ratio': 0.71
                },
                {
                    'name': 'Equal Weight',
                    'expected_return': 0.128,
                    'expected_volatility': 0.25,
                    'sharpe_ratio': 0.65
                },
                {
                    'name': 'Momentum',
                    'expected_return': 0.156,
                    'expected_volatility': 0.28,
                    'sharpe_ratio': 0.69
                }
            ],
            'sensitivity_analysis': {
                'return_sensitivity': {
                    'conservative': {'return': 0.125, 'volatility': 0.18, 'sharpe': 0.72},
                    'moderate': {'return': 0.145, 'volatility': 0.22, 'sharpe': 0.76},
                    'aggressive': {'return': 0.168, 'volatility': 0.28, 'sharpe': 0.73}
                }
            }
        }
        
        return jsonify({
            'success': True,
            'data': optimization_results,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@portfolio_bp.route('/api/portfolio/rebalance/suggestions', methods=['POST'])
@cross_origin()
def get_rebalance_suggestions():
    """
    Endpoint para sugestões de rebalanceamento.
    """
    try:
        data = request.get_json()
        
        current_portfolio = data.get('current_portfolio', {})
        target_allocation = data.get('target_allocation', 'risk_parity')
        rebalance_threshold = data.get('threshold', 0.05)  # 5%
        
        # Análise de rebalanceamento
        rebalance_analysis = {
            'analysis_date': datetime.now().isoformat(),
            'current_allocation': current_portfolio,
            'target_strategy': target_allocation,
            'drift_analysis': {
                'total_drift': 0.08,
                'assets_out_of_range': ['PETR4', 'ITUB4', 'WEGE3'],
                'max_drift_asset': 'PETR4',
                'max_drift_amount': 0.032
            },
            'rebalance_recommendations': [
                {
                    'ticker': 'PETR4',
                    'current_weight': 0.112,
                    'target_weight': 0.080,
                    'drift': 0.032,
                    'action': 'REDUCE',
                    'amount_to_trade': -4650.00,
                    'urgency': 'HIGH'
                },
                {
                    'ticker': 'ITUB4',
                    'current_weight': 0.150,
                    'target_weight': 0.120,
                    'drift': 0.030,
                    'action': 'REDUCE',
                    'amount_to_trade': -4357.00,
                    'urgency': 'MEDIUM'
                },
                {
                    'ticker': 'WEGE3',
                    'current_weight': 0.100,
                    'target_weight': 0.130,
                    'drift': -0.030,
                    'action': 'INCREASE',
                    'amount_to_trade': 4357.00,
                    'urgency': 'MEDIUM'
                }
            ],
            'cost_analysis': {
                'total_transaction_cost': 520.00,
                'cost_as_percentage': 0.36,
                'break_even_period': '3 months',
                'net_benefit_12m': 3200.00
            },
            'timing_analysis': {
                'recommended_timing': 'Within 2 weeks',
                'market_conditions': 'Favorable',
                'volatility_forecast': 'Normal',
                'liquidity_conditions': 'Good'
            },
            'tax_implications': {
                'estimated_tax_impact': 450.00,
                'tax_loss_harvesting_opportunities': ['BBDC4'],
                'holding_period_considerations': 'Some positions < 1 year'
            }
        }
        
        return jsonify({
            'success': True,
            'data': rebalance_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@portfolio_bp.route('/api/portfolio/simulation', methods=['POST'])
@cross_origin()
def portfolio_simulation():
    """
    Endpoint para simulação Monte Carlo do portfólio.
    """
    try:
        data = request.get_json()
        
        portfolio_weights = data.get('portfolio_weights', {})
        time_horizon = data.get('time_horizon', 252)  # 1 ano
        num_simulations = data.get('num_simulations', 1000)
        initial_value = data.get('initial_value', 100000)
        
        # Simulação Monte Carlo
        np.random.seed(42)
        
        # Parâmetros simulados
        mean_returns = np.array([0.12, 0.10, 0.08, 0.15, 0.09])  # retornos anualizados
        volatilities = np.array([0.28, 0.32, 0.22, 0.18, 0.24])  # volatilidades anualizadas
        
        # Simulação de caminhos
        simulation_results = []
        final_values = []
        
        for _ in range(num_simulations):
            portfolio_path = [initial_value]
            current_value = initial_value
            
            for day in range(time_horizon):
                # Retorno diário simulado
                daily_return = np.random.normal(
                    np.mean(mean_returns) / 252,  # retorno médio diário
                    np.mean(volatilities) / np.sqrt(252)  # volatilidade diária
                )
                current_value *= (1 + daily_return)
                portfolio_path.append(current_value)
            
            final_values.append(current_value)
            
            # Armazena apenas algumas simulações para visualização
            if len(simulation_results) < 10:
                simulation_results.append(portfolio_path)
        
        final_values = np.array(final_values)
        
        # Estatísticas da simulação
        simulation_analysis = {
            'simulation_parameters': {
                'num_simulations': num_simulations,
                'time_horizon_days': time_horizon,
                'initial_value': initial_value
            },
            'final_value_statistics': {
                'mean': float(np.mean(final_values)),
                'median': float(np.median(final_values)),
                'std': float(np.std(final_values)),
                'min': float(np.min(final_values)),
                'max': float(np.max(final_values)),
                'percentile_5': float(np.percentile(final_values, 5)),
                'percentile_25': float(np.percentile(final_values, 25)),
                'percentile_75': float(np.percentile(final_values, 75)),
                'percentile_95': float(np.percentile(final_values, 95))
            },
            'return_statistics': {
                'mean_return': float((np.mean(final_values) / initial_value - 1) * 100),
                'probability_of_loss': float(np.sum(final_values < initial_value) / num_simulations * 100),
                'probability_positive_return': float(np.sum(final_values > initial_value) / num_simulations * 100),
                'probability_double_money': float(np.sum(final_values > 2 * initial_value) / num_simulations * 100)
            },
            'risk_metrics': {
                'var_95': float(np.percentile(final_values, 5) - initial_value),
                'cvar_95': float(np.mean(final_values[final_values <= np.percentile(final_values, 5)]) - initial_value),
                'maximum_loss': float(np.min(final_values) - initial_value),
                'volatility_of_returns': float(np.std(final_values / initial_value - 1) * 100)
            },
            'sample_paths': simulation_results[:5]  # Primeiros 5 caminhos para visualização
        }
        
        return jsonify({
            'success': True,
            'data': simulation_analysis,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@portfolio_bp.route('/api/portfolio/reports/generate', methods=['POST'])
@cross_origin()
def generate_portfolio_report():
    """
    Endpoint para geração de relatório completo do portfólio.
    """
    try:
        data = request.get_json()
        
        report_type = data.get('report_type', 'comprehensive')
        period = data.get('period', '1Y')
        include_sections = data.get('sections', ['summary', 'performance', 'risk', 'recommendations'])
        
        # Estrutura do relatório
        portfolio_report = {
            'report_metadata': {
                'generated_at': datetime.now().isoformat(),
                'report_type': report_type,
                'period_analyzed': period,
                'user_id': current_user.id,
                'report_version': '2.0'
            },
            'executive_summary': {
                'portfolio_value': 145230.50,
                'total_return': 21.03,
                'risk_adjusted_return': 0.68,  # Sharpe ratio
                'key_highlights': [
                    'Portfólio superou benchmark em 8.5pp no período',
                    'Diversificação adequada com exposição a 5 setores',
                    'Risco controlado com VaR-95% de -4.8%',
                    'Oportunidades de otimização identificadas'
                ],
                'main_concerns': [
                    'Concentração elevada no setor financeiro (26.2%)',
                    'Beta acima de 1.0 indica maior volatilidade',
                    'Algumas posições podem se beneficiar de rebalanceamento'
                ]
            },
            'performance_analysis': {
                'absolute_performance': {
                    '1M': 5.8,
                    '3M': 12.1,
                    '6M': 18.7,
                    '1Y': 21.0
                },
                'relative_performance': {
                    'vs_ibovespa': 8.5,
                    'vs_peers': 4.2,
                    'vs_risk_free': 10.5
                },
                'risk_metrics': {
                    'volatility': 24.0,
                    'sharpe_ratio': 0.68,
                    'max_drawdown': -18.0,
                    'var_95': -4.8
                }
            },
            'asset_analysis': [
                {
                    'ticker': 'PETR4',
                    'performance_contribution': 1.95,
                    'risk_contribution': 18.0,
                    'recommendation': 'REDUCE'
                },
                {
                    'ticker': 'WEGE3',
                    'performance_contribution': 1.95,
                    'risk_contribution': 12.0,
                    'recommendation': 'INCREASE'
                }
            ],
            'recommendations': {
                'immediate_actions': [
                    'Reduzir exposição a PETR4 de 11.2% para 8%',
                    'Aumentar posição em WEGE3 para 15%',
                    'Considerar adição de ABEV3 ao portfólio'
                ],
                'medium_term_strategy': [
                    'Diversificar geograficamente com ETFs internacionais',
                    'Incluir instrumentos de renda fixa para reduzir volatilidade',
                    'Implementar estratégia de dividend growth'
                ],
                'risk_management': [
                    'Estabelecer stop-loss em 15% para posições individuais',
                    'Monitorar correlação com Ibovespa',
                    'Revisar portfólio mensalmente'
                ]
            },
            'appendix': {
                'methodology': 'Análise baseada em métricas quantitativas e teoria moderna de portfólio',
                'data_sources': 'B3, Yahoo Finance, Fundamentus',
                'assumptions': 'Mercado eficiente, distribuição normal dos retornos',
                'disclaimer': 'Este relatório não constitui recomendação de investimento'
            }
        }
        
        return jsonify({
            'success': True,
            'data': portfolio_report,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500
