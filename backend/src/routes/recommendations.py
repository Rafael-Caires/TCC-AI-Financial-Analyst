"""
Rotas para recomendações de investimento

Este arquivo implementa as rotas para recomendações de investimento
baseadas em previsões de modelos de ML e perfil do usuário.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

from flask import Blueprint, request, jsonify, current_app
from flask_cors import cross_origin
from src.services.financial_data_service import FinancialDataService
from src.services.recommendation_service import RecommendationService
from src.ml.model_integrator import ModelIntegrator
from src.utils.auth import token_required
from src.models.models import User, Portfolio, PortfolioAsset, Watchlist
import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta

# Cria o blueprint para as rotas de recomendações
recommendations_bp = Blueprint('recommendations', __name__, url_prefix='/api/recommendations')

# Inicializa o integrador de modelos
model_integrator = ModelIntegrator(models_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models'))

# Função para carregar usuário pelo ID
def load_user_by_id(user_id):
    """
    Carrega um usuário pelo ID.
    
    Args:
        user_id (int): ID do usuário
        
    Returns:
        User: Objeto do usuário ou None se não encontrado
    """
    # Caminho para o arquivo de usuários
    users_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'users.json')
    
    if not os.path.exists(users_file):
        return None
    
    try:
        with open(users_file, 'r') as f:
            users_data = json.load(f)
        
        for user_data in users_data:
            if user_data.get('id') == user_id:
                return User.from_dict(user_data)
        
        return None
    except Exception as e:
        current_app.logger.error(f"Erro ao carregar usuário: {str(e)}")
        return None

# Função para carregar portfólio do usuário
def load_user_portfolio(user_id):
    """
    Carrega o portfólio de um usuário.
    
    Args:
        user_id (int): ID do usuário
        
    Returns:
        Portfolio: Objeto do portfólio ou None se não encontrado
    """
    # Caminho para o arquivo de portfólios
    portfolios_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'portfolios.json')
    
    if not os.path.exists(portfolios_file):
        # Cria um portfólio padrão
        return create_default_portfolio(user_id)
    
    try:
        with open(portfolios_file, 'r') as f:
            portfolios_data = json.load(f)
        
        for portfolio_data in portfolios_data:
            if portfolio_data.get('user_id') == user_id:
                return Portfolio.from_dict(portfolio_data)
        
        # Se não encontrou, cria um portfólio padrão
        return create_default_portfolio(user_id)
    except Exception as e:
        current_app.logger.error(f"Erro ao carregar portfólio: {str(e)}")
        return create_default_portfolio(user_id)

# Função para criar um portfólio padrão
def create_default_portfolio(user_id):
    """
    Cria um portfólio padrão para um usuário.
    
    Args:
        user_id (int): ID do usuário
        
    Returns:
        Portfolio: Objeto do portfólio padrão
    """
    portfolio = Portfolio(
        id=1,
        user_id=user_id,
        name="Meu Portfólio",
        description="Portfólio padrão"
    )
    
    # Adiciona alguns ativos de exemplo
    portfolio.add_asset(PortfolioAsset(
        id=1,
        portfolio_id=1,
        ticker="PETR4.SA",
        quantity=100,
        purchase_price=30.0,
        purchase_date=datetime.now() - timedelta(days=30)
    ))
    
    portfolio.add_asset(PortfolioAsset(
        id=2,
        portfolio_id=1,
        ticker="VALE3.SA",
        quantity=50,
        purchase_price=70.0,
        purchase_date=datetime.now() - timedelta(days=60)
    ))
    
    # Salva o portfólio
    save_portfolio(portfolio)
    
    return portfolio

# Função para salvar um portfólio
def save_portfolio(portfolio):
    """
    Salva um portfólio no arquivo.
    
    Args:
        portfolio (Portfolio): Objeto do portfólio
    """
    # Caminho para o arquivo de portfólios
    portfolios_file = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'portfolios.json')
    
    # Garante que o diretório existe
    os.makedirs(os.path.dirname(portfolios_file), exist_ok=True)
    
    try:
        # Carrega portfólios existentes
        portfolios = []
        if os.path.exists(portfolios_file):
            with open(portfolios_file, 'r') as f:
                portfolios_data = json.load(f)
                for p_data in portfolios_data:
                    if p_data.get('id') != portfolio.id:
                        portfolios.append(p_data)
        
        # Adiciona o portfólio atualizado
        portfolios.append(portfolio.to_dict())
        
        # Salva no arquivo
        with open(portfolios_file, 'w') as f:
            json.dump(portfolios, f, indent=2)
    except Exception as e:
        current_app.logger.error(f"Erro ao salvar portfólio: {str(e)}")

# Rota para obter recomendação para um ativo
@recommendations_bp.route('/stock', methods=['GET'])
@token_required
def get_stock_recommendation(user_id):
    """
    Obtém recomendação para um ativo específico.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Recomendação para o ativo
    """
    # Obtém parâmetros da requisição
    ticker = request.args.get('ticker')
    
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker não fornecido'}), 400
    
    try:
        # Carrega o usuário para obter o perfil de risco
        user = load_user_by_id(user_id)
        if not user:
            return jsonify({'success': False, 'message': 'Usuário não encontrado'}), 404
        
        # Obtém dados históricos
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        data = FinancialDataService.get_stock_data(ticker, start_date=start_date)
        
        # Verifica se há dados suficientes
        if len(data) < 252:  # Pelo menos 1 ano de dados
            return jsonify({
                'success': False, 
                'message': 'Não há dados históricos suficientes para fazer recomendações'
            }), 400
        
        # Carrega os modelos
        try:
            model_integrator.load_all_models()
        except Exception as e:
            current_app.logger.warning(f"Erro ao carregar modelos: {str(e)}. Treinando novos modelos...")
            # Se não conseguir carregar, treina novos modelos
            model_integrator.train_all_models(data, verbose=0)
            model_integrator.save_all_models()
        
        # Faz a previsão com o ensemble
        prediction = model_integrator.predict(data, days_ahead=30, use_ensemble=True)
        
        # Gera a recomendação
        recommendation = RecommendationService.generate_stock_recommendation(
            prediction,
            user.risk_profile,
            current_price=data['Close'].iloc[-1]
        )
        
        return jsonify({
            'success': True,
            'recommendation': recommendation
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao gerar recomendação: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao gerar recomendação: {str(e)}'}), 500

# Rota para obter recomendações para o portfólio
@recommendations_bp.route('/portfolio', methods=['GET'])
@token_required
def get_portfolio_recommendations(user_id):
    """
    Obtém recomendações para o portfólio do usuário.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Recomendações para o portfólio
    """
    try:
        # Carrega o usuário para obter o perfil de risco
        user = load_user_by_id(user_id)
        if not user:
            return jsonify({'success': False, 'message': 'Usuário não encontrado'}), 404
        
        # Carrega o portfólio do usuário
        portfolio = load_user_portfolio(user_id)
        if not portfolio or not portfolio.assets:
            return jsonify({
                'success': False, 
                'message': 'Portfólio vazio ou não encontrado'
            }), 404
        
        # Prepara os ativos do portfólio
        portfolio_assets = []
        for asset in portfolio.assets:
            portfolio_assets.append({
                'ticker': asset.ticker,
                'quantity': asset.quantity,
                'purchase_price': asset.purchase_price,
                'purchase_date': asset.purchase_date.isoformat() if asset.purchase_date else None
            })
        
        # Carrega os modelos
        try:
            model_integrator.load_all_models()
        except Exception as e:
            current_app.logger.warning(f"Erro ao carregar modelos: {str(e)}. Os modelos serão treinados para cada ativo.")
        
        # Faz previsões para cada ativo do portfólio
        predictions = {}
        
        for asset in portfolio_assets:
            ticker = asset['ticker']
            try:
                # Obtém dados históricos
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                data = FinancialDataService.get_stock_data(ticker, start_date=start_date)
                
                # Verifica se há dados suficientes
                if len(data) < 252:  # Pelo menos 1 ano de dados
                    continue
                
                # Treina modelos se necessário
                if not all(model is not None for model in model_integrator.models.values()):
                    model_integrator.train_all_models(data, verbose=0)
                    model_integrator.save_all_models()
                
                # Faz a previsão com o ensemble
                prediction = model_integrator.predict(data, days_ahead=30, use_ensemble=True)
                
                # Adiciona o preço atual
                asset['current_price'] = data['Close'].iloc[-1]
                
                # Adiciona à lista de previsões
                predictions[ticker] = prediction
            
            except Exception as e:
                current_app.logger.error(f"Erro ao gerar previsão para {ticker}: {str(e)}")
                continue
        
        # Gera recomendações para o portfólio
        portfolio_recommendations = RecommendationService.generate_portfolio_recommendations(
            portfolio_assets,
            predictions,
            user.risk_profile
        )
        
        return jsonify({
            'success': True,
            'portfolio': portfolio.to_dict(),
            'recommendations': portfolio_recommendations
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao gerar recomendações para o portfólio: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao gerar recomendações: {str(e)}'}), 500

# Rota para obter recomendações de mercado
@recommendations_bp.route('/market', methods=['GET'])
@token_required
def get_market_recommendations(user_id):
    """
    Obtém recomendações gerais de mercado.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Recomendações de mercado
    """
    try:
        # Carrega o usuário para obter o perfil de risco
        user = load_user_by_id(user_id)
        if not user:
            return jsonify({'success': False, 'message': 'Usuário não encontrado'}), 404
        
        # Obtém dados de mercado
        market_summary = FinancialDataService.get_market_summary()
        sector_performance = FinancialDataService.get_sector_performance()
        
        market_data = {
            'indices': market_summary,
            'sectors': sector_performance
        }
        
        # Gera recomendações de mercado
        market_recommendations = RecommendationService.generate_market_recommendations(
            market_data,
            user.risk_profile
        )
        
        return jsonify({
            'success': True,
            'market_data': market_data,
            'recommendations': market_recommendations
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao gerar recomendações de mercado: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao gerar recomendações: {str(e)}'}), 500

# Rota para obter recomendações para múltiplos ativos
@recommendations_bp.route('/batch', methods=['POST'])
@token_required
def batch_recommendations(user_id):
    """
    Obtém recomendações para múltiplos ativos.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Recomendações para os ativos
    """
    data = request.get_json()
    
    # Obtém parâmetros
    tickers = data.get('tickers', [])
    
    if not tickers:
        return jsonify({'success': False, 'message': 'Nenhum ticker fornecido'}), 400
    
    try:
        # Carrega o usuário para obter o perfil de risco
        user = load_user_by_id(user_id)
        if not user:
            return jsonify({'success': False, 'message': 'Usuário não encontrado'}), 404
        
        # Carrega os modelos
        try:
            model_integrator.load_all_models()
        except Exception as e:
            current_app.logger.warning(f"Erro ao carregar modelos: {str(e)}. Os modelos serão treinados para cada ativo.")
        
        # Faz recomendações para cada ativo
        results = {}
        
        for ticker in tickers:
            try:
                # Obtém dados históricos
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                stock_data = FinancialDataService.get_stock_data(ticker, start_date=start_date)
                
                # Verifica se há dados suficientes
                if len(stock_data) < 252:  # Pelo menos 1 ano de dados
                    results[ticker] = {
                        'success': False,
                        'message': 'Dados históricos insuficientes'
                    }
                    continue
                
                # Treina modelos se necessário
                if not all(model is not None for model in model_integrator.models.values()):
                    model_integrator.train_all_models(stock_data, verbose=0)
                    model_integrator.save_all_models()
                
                # Faz a previsão com o ensemble
                prediction = model_integrator.predict(stock_data, days_ahead=30, use_ensemble=True)
                
                # Gera a recomendação
                recommendation = RecommendationService.generate_stock_recommendation(
                    prediction,
                    user.risk_profile,
                    current_price=stock_data['Close'].iloc[-1]
                )
                
                # Adiciona à resposta
                results[ticker] = {
                    'success': True,
                    'recommendation': recommendation
                }
            
            except Exception as e:
                current_app.logger.error(f"Erro ao gerar recomendação para {ticker}: {str(e)}")
                results[ticker] = {
                    'success': False,
                    'message': f'Erro: {str(e)}'
                }
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao processar lote de recomendações: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao processar lote: {str(e)}'}), 500

# Rota para atualizar o portfólio
@recommendations_bp.route('/portfolio', methods=['POST'])
@token_required
def update_portfolio(user_id):
    """
    Atualiza o portfólio do usuário.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Status da atualização
    """
    data = request.get_json()
    
    try:
        # Carrega o portfólio atual
        portfolio = load_user_portfolio(user_id)
        
        # Atualiza os campos básicos
        if 'name' in data:
            portfolio.name = data['name']
        
        if 'description' in data:
            portfolio.description = data['description']
        
        # Atualiza os ativos
        if 'assets' in data:
            # Limpa os ativos atuais
            portfolio.assets = []
            
            # Adiciona os novos ativos
            for i, asset_data in enumerate(data['assets']):
                asset = PortfolioAsset(
                    id=i + 1,
                    portfolio_id=portfolio.id,
                    ticker=asset_data.get('ticker'),
                    quantity=asset_data.get('quantity'),
                    purchase_price=asset_data.get('purchase_price'),
                    purchase_date=datetime.fromisoformat(asset_data['purchase_date']) if 'purchase_date' in asset_data else datetime.now()
                )
                portfolio.add_asset(asset)
        
        # Salva o portfólio atualizado
        save_portfolio(portfolio)
        
        return jsonify({
            'success': True,
            'message': 'Portfólio atualizado com sucesso',
            'portfolio': portfolio.to_dict()
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao atualizar portfólio: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao atualizar portfólio: {str(e)}'}), 500

# Rota para recomendações avançadas (sistema híbrido)
@recommendations_bp.route('/advanced', methods=['GET', 'POST'])
@cross_origin()
def get_advanced_recommendations():
    """
    Obtém recomendações avançadas usando sistema híbrido.
    
    Returns:
        JSON: Recomendações avançadas
    """
    try:
        # Parâmetros da requisição
        limit = int(request.args.get('limit', 20))
        risk_level = request.args.get('risk_level', 'medium')
        investment_goal = request.args.get('investment_goal', 'growth')
        time_horizon = request.args.get('time_horizon', 'medium_term')
        
        # Usa dados simulados para desenvolvimento
        user_id = 1  # Usuário padrão para desenvolvimento
        
        # Sistema híbrido de recomendações com dados simulados avançados
        recommendations = {
            'hybrid_recommendations': [
                {
                    'ticker': 'WEGE3',
                    'name': 'WEG ON',
                    'sector': 'Industrial',
                    'current_price': 48.50,
                    'target_price': 55.20,
                    'potential_return': 13.8,
                    'risk_score': 0.55,
                    'recommendation': 'BUY',
                    'confidence': 0.87,
                    'hybrid_score': 8.5,
                    'analysis': {
                        'collaborative_score': 8.2,
                        'content_based_score': 8.8,
                        'sentiment_score': 7.9,
                        'technical_score': 8.7,
                        'fundamental_score': 8.4
                    },
                    'reasons': [
                        'Forte crescimento em exportações',
                        'Liderança em motores elétricos',
                        'Indicadores técnicos bullish',
                        'Sentiment positivo nas notícias'
                    ]
                },
                {
                    'ticker': 'RENT3',
                    'name': 'Localiza Rent a Car ON',
                    'sector': 'Serviços',
                    'current_price': 62.80,
                    'target_price': 72.50,
                    'potential_return': 15.4,
                    'risk_score': 0.62,
                    'recommendation': 'BUY',
                    'confidence': 0.82,
                    'hybrid_score': 8.3,
                    'analysis': {
                        'collaborative_score': 8.1,
                        'content_based_score': 8.6,
                        'sentiment_score': 8.2,
                        'technical_score': 8.0,
                        'fundamental_score': 8.5
                    },
                    'reasons': [
                        'Recuperação do setor de turismo',
                        'Expansão da frota elétrica',
                        'Múltiplos atrativos',
                        'Cash flow positivo'
                    ]
                },
                {
                    'ticker': 'MGLU3',
                    'name': 'Magazine Luiza ON',
                    'sector': 'Comércio',
                    'current_price': 14.25,
                    'target_price': 18.80,
                    'potential_return': 31.9,
                    'risk_score': 0.78,
                    'recommendation': 'SPECULATIVE_BUY',
                    'confidence': 0.65,
                    'hybrid_score': 7.8,
                    'analysis': {
                        'collaborative_score': 7.5,
                        'content_based_score': 8.2,
                        'sentiment_score': 7.1,
                        'technical_score': 7.8,
                        'fundamental_score': 8.4
                    },
                    'reasons': [
                        'Recuperação pós-correção',
                        'Inovação em marketplace',
                        'Melhoria na margem',
                        'Potencial de reversão'
                    ]
                },
                {
                    'ticker': 'GGBR4',
                    'name': 'Gerdau PN',
                    'sector': 'Siderurgia',
                    'current_price': 23.45,
                    'target_price': 28.20,
                    'potential_return': 20.3,
                    'risk_score': 0.71,
                    'recommendation': 'BUY',
                    'confidence': 0.79,
                    'hybrid_score': 8.1,
                    'analysis': {
                        'collaborative_score': 8.0,
                        'content_based_score': 8.3,
                        'sentiment_score': 7.8,
                        'technical_score': 8.2,
                        'fundamental_score': 8.2
                    },
                    'reasons': [
                        'Recuperação da construção civil',
                        'Melhoria nos preços do aço',
                        'Gestão eficiente de custos',
                        'Dividendos atrativos'
                    ]
                },
                {
                    'ticker': 'SUZB3',
                    'name': 'Suzano ON',
                    'sector': 'Papel e Celulose',
                    'current_price': 54.20,
                    'target_price': 62.80,
                    'potential_return': 15.9,
                    'risk_score': 0.69,
                    'recommendation': 'BUY',
                    'confidence': 0.84,
                    'hybrid_score': 8.4,
                    'analysis': {
                        'collaborative_score': 8.3,
                        'content_based_score': 8.5,
                        'sentiment_score': 8.1,
                        'technical_score': 8.6,
                        'fundamental_score': 8.5
                    },
                    'reasons': [
                        'Preços internacionais em alta',
                        'Sustentabilidade como vantagem',
                        'Operações eficientes',
                        'Demanda chinesa forte'
                    ]
                },
                {
                    'ticker': 'RADL3',
                    'name': 'RaiaDrogasil ON',
                    'sector': 'Farmácia e Higiene',
                    'current_price': 28.90,
                    'target_price': 34.50,
                    'potential_return': 19.4,
                    'risk_score': 0.58,
                    'recommendation': 'BUY',
                    'confidence': 0.81,
                    'hybrid_score': 8.2,
                    'analysis': {
                        'collaborative_score': 8.1,
                        'content_based_score': 8.4,
                        'sentiment_score': 8.0,
                        'technical_score': 8.3,
                        'fundamental_score': 8.2
                    },
                    'reasons': [
                        'Expansão de lojas',
                        'Margem em melhoria',
                        'Setor defensivo',
                        'Market share crescente'
                    ]
                }
            ],
            'market_analysis': {
                'current_regime': 'Consolidação',
                'sentiment': 'Neutro positivo',
                'volatility_forecast': 'Média',
                'key_themes': [
                    'Taxa Selic em queda',
                    'Recuperação gradual da economia',
                    'Eleições americanas',
                    'Política fiscal brasileira'
                ]
            },
            'sector_allocation_suggestion': [
                {'sector': 'Industrial', 'recommended_weight': 25, 'current_market_weight': 18},
                {'sector': 'Financeiro', 'recommended_weight': 20, 'current_market_weight': 28},
                {'sector': 'Commodities', 'recommended_weight': 15, 'current_market_weight': 22},
                {'sector': 'Consumo', 'recommended_weight': 20, 'current_market_weight': 16},
                {'sector': 'Tecnologia', 'recommended_weight': 10, 'current_market_weight': 8},
                {'sector': 'Outros', 'recommended_weight': 10, 'current_market_weight': 8}
            ],
            'risk_analysis': {
                'portfolio_var_95': -4.2,
                'recommended_diversification': 8,
                'correlation_with_ibovespa': 0.75,
                'maximum_single_position': 15
            },
            'filters_applied': {
                'risk_level': risk_level,
                'investment_goal': investment_goal,
                'time_horizon': time_horizon,
                'user_preferences': 'medium'  # Simulado para desenvolvimento
            }
        }
        
        # Aplica limite
        recommendations['hybrid_recommendations'] = recommendations['hybrid_recommendations'][:limit]
        
        return jsonify({
            'success': True,
            'data': recommendations,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao gerar recomendações avançadas: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao gerar recomendações: {str(e)}'}), 500
