"""
Rotas para dados de ações e análise financeira

Este arquivo implementa as rotas para obtenção e análise de dados financeiros.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

from flask import Blueprint, request, jsonify, current_app
from src.services.financial_data_service import FinancialDataService
from src.utils.auth import token_required
import pandas as pd
import json
import os
import base64
from datetime import datetime, timedelta

# Cria o blueprint para as rotas de ações
stocks_bp = Blueprint('stocks', __name__, url_prefix='/api/stocks')

# Rota para obter dados históricos
@stocks_bp.route('/historical', methods=['GET'])
@token_required
def get_historical_data(user_id):
    """
    Obtém dados históricos de um ativo.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Dados históricos do ativo
    """
    # Obtém parâmetros da requisição
    ticker = request.args.get('ticker')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    period = request.args.get('period')
    
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker não fornecido'}), 400
    
    try:
        # Obtém dados históricos
        data = FinancialDataService.get_stock_data(ticker, start_date, end_date, period)
        
        # Verifica se há dados
        if data.empty:
            return jsonify({'success': False, 'message': 'Não foram encontrados dados para o ticker informado'}), 404
        
        # Converte para JSON
        data_json = data.reset_index().to_json(orient='records', date_format='iso')
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'data': data_json
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao obter dados históricos: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao obter dados: {str(e)}'}), 500

# Rota para obter dados com indicadores técnicos
@stocks_bp.route('/indicators', methods=['GET'])
@token_required
def get_technical_indicators(user_id):
    """
    Obtém dados com indicadores técnicos de um ativo.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Dados com indicadores técnicos
    """
    # Obtém parâmetros da requisição
    ticker = request.args.get('ticker')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    period = request.args.get('period')
    
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker não fornecido'}), 400
    
    try:
        # Obtém dados históricos
        data = FinancialDataService.get_stock_data(ticker, start_date, end_date, period)
        
        # Verifica se há dados
        if data.empty:
            return jsonify({'success': False, 'message': 'Não foram encontrados dados para o ticker informado'}), 404
        
        # Calcula indicadores técnicos
        data_with_indicators = FinancialDataService.calculate_technical_indicators(data)
        
        # Converte para JSON
        data_json = data_with_indicators.reset_index().to_json(orient='records', date_format='iso')
        
        # Calcula retornos e volatilidade
        returns = FinancialDataService.calculate_returns(data)
        volatility = FinancialDataService.calculate_volatility(data)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'data': data_json,
            'returns': returns,
            'volatility': volatility
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao obter indicadores técnicos: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao obter indicadores: {str(e)}'}), 500

# Rota para obter gráfico de preços
@stocks_bp.route('/price-chart', methods=['GET'])
@token_required
def get_price_chart(user_id):
    """
    Obtém um gráfico de preços de um ativo.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Imagem do gráfico em base64
    """
    # Obtém parâmetros da requisição
    ticker = request.args.get('ticker')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    period = request.args.get('period')
    indicators = request.args.get('indicators', '').split(',') if request.args.get('indicators') else None
    
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker não fornecido'}), 400
    
    try:
        # Obtém dados históricos
        data = FinancialDataService.get_stock_data(ticker, start_date, end_date, period)
        
        # Verifica se há dados
        if data.empty:
            return jsonify({'success': False, 'message': 'Não foram encontrados dados para o ticker informado'}), 404
        
        # Calcula indicadores técnicos se necessário
        if indicators:
            data = FinancialDataService.calculate_technical_indicators(data)
        
        # Gera o gráfico
        chart_base64 = FinancialDataService.generate_price_chart(data, ticker, indicators)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'chart': chart_base64
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao gerar gráfico: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao gerar gráfico: {str(e)}'}), 500

# Rota para obter gráfico de indicador técnico
@stocks_bp.route('/technical-chart', methods=['GET'])
@token_required
def get_technical_chart(user_id):
    """
    Obtém um gráfico de indicador técnico de um ativo.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Imagem do gráfico em base64
    """
    # Obtém parâmetros da requisição
    ticker = request.args.get('ticker')
    indicator = request.args.get('indicator')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    period = request.args.get('period')
    
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker não fornecido'}), 400
    
    if not indicator:
        return jsonify({'success': False, 'message': 'Indicador não fornecido'}), 400
    
    try:
        # Obtém dados históricos
        data = FinancialDataService.get_stock_data(ticker, start_date, end_date, period)
        
        # Verifica se há dados
        if data.empty:
            return jsonify({'success': False, 'message': 'Não foram encontrados dados para o ticker informado'}), 404
        
        # Calcula indicadores técnicos
        data = FinancialDataService.calculate_technical_indicators(data)
        
        # Gera o gráfico
        chart_base64 = FinancialDataService.generate_technical_chart(data, ticker, indicator)
        
        return jsonify({
            'success': True,
            'ticker': ticker,
            'indicator': indicator,
            'chart': chart_base64
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao gerar gráfico técnico: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao gerar gráfico técnico: {str(e)}'}), 500

# Rota para obter informações de um ativo
@stocks_bp.route('/info', methods=['GET'])
@token_required
def get_stock_info(user_id):
    """
    Obtém informações gerais sobre um ativo.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Informações do ativo
    """
    # Obtém parâmetros da requisição
    ticker = request.args.get('ticker')
    
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker não fornecido'}), 400
    
    try:
        # Obtém informações do ativo
        info = FinancialDataService.get_stock_info(ticker)
        
        # Obtém dados históricos recentes para complementar
        data = FinancialDataService.get_stock_data(ticker, period='1mo')
        
        # Verifica se há dados
        if data.empty:
            return jsonify({'success': False, 'message': 'Não foram encontrados dados para o ticker informado'}), 404
        
        # Calcula retornos e volatilidade
        returns = FinancialDataService.calculate_returns(data)
        volatility = FinancialDataService.calculate_volatility(data)
        
        # Adiciona informações complementares
        info['current_price'] = data['Close'].iloc[-1]
        info['price_change'] = ((data['Close'].iloc[-1] / data['Close'].iloc[-2]) - 1) * 100
        info['returns'] = returns
        info['volatility'] = volatility
        
        return jsonify({
            'success': True,
            'info': info
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao obter informações: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao obter informações: {str(e)}'}), 500

# Rota para obter resumo do mercado
@stocks_bp.route('/market-summary', methods=['GET'])
@token_required
def get_market_summary(user_id):
    """
    Obtém um resumo do mercado.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Resumo do mercado
    """
    try:
        # Obtém resumo do mercado
        summary = FinancialDataService.get_market_summary()
        
        # Obtém desempenho por setor
        sectors = FinancialDataService.get_sector_performance()
        
        return jsonify({
            'success': True,
            'indices': summary,
            'sectors': sectors,
            'timestamp': datetime.now().isoformat()
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao obter resumo do mercado: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao obter resumo do mercado: {str(e)}'}), 500

# Rota para buscar ativos
@stocks_bp.route('/search', methods=['GET'])
@token_required
def search_stocks(user_id):
    """
    Busca ativos pelo nome ou símbolo.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Lista de ativos encontrados
    """
    # Obtém parâmetros da requisição
    query = request.args.get('query')
    
    if not query or len(query) < 2:
        return jsonify({'success': False, 'message': 'Termo de busca muito curto'}), 400
    
    try:
        # Lista de ativos brasileiros mais comuns (simulação)
        br_stocks = [
            {'ticker': 'PETR4.SA', 'name': 'Petrobras PN', 'sector': 'Petróleo e Gás'},
            {'ticker': 'VALE3.SA', 'name': 'Vale ON', 'sector': 'Mineração'},
            {'ticker': 'ITUB4.SA', 'name': 'Itaú Unibanco PN', 'sector': 'Financeiro'},
            {'ticker': 'BBDC4.SA', 'name': 'Bradesco PN', 'sector': 'Financeiro'},
            {'ticker': 'ABEV3.SA', 'name': 'Ambev ON', 'sector': 'Bebidas'},
            {'ticker': 'MGLU3.SA', 'name': 'Magazine Luiza ON', 'sector': 'Varejo'},
            {'ticker': 'WEGE3.SA', 'name': 'WEG ON', 'sector': 'Bens Industriais'},
            {'ticker': 'RENT3.SA', 'name': 'Localiza ON', 'sector': 'Aluguel de Veículos'},
            {'ticker': 'BBAS3.SA', 'name': 'Banco do Brasil ON', 'sector': 'Financeiro'},
            {'ticker': 'ITSA4.SA', 'name': 'Itaúsa PN', 'sector': 'Holding'},
            {'ticker': 'B3SA3.SA', 'name': 'B3 ON', 'sector': 'Financeiro'},
            {'ticker': 'RADL3.SA', 'name': 'Raia Drogasil ON', 'sector': 'Saúde'},
            {'ticker': 'SUZB3.SA', 'name': 'Suzano ON', 'sector': 'Papel e Celulose'},
            {'ticker': 'JBSS3.SA', 'name': 'JBS ON', 'sector': 'Alimentos'},
            {'ticker': 'LREN3.SA', 'name': 'Lojas Renner ON', 'sector': 'Varejo'}
        ]
        
        # Lista de ativos americanos mais comuns (simulação)
        us_stocks = [
            {'ticker': 'AAPL', 'name': 'Apple Inc.', 'sector': 'Tecnologia'},
            {'ticker': 'MSFT', 'name': 'Microsoft Corporation', 'sector': 'Tecnologia'},
            {'ticker': 'AMZN', 'name': 'Amazon.com Inc.', 'sector': 'Varejo/Tecnologia'},
            {'ticker': 'GOOGL', 'name': 'Alphabet Inc. (Google)', 'sector': 'Tecnologia'},
            {'ticker': 'META', 'name': 'Meta Platforms Inc. (Facebook)', 'sector': 'Tecnologia'},
            {'ticker': 'TSLA', 'name': 'Tesla Inc.', 'sector': 'Automóveis'},
            {'ticker': 'NVDA', 'name': 'NVIDIA Corporation', 'sector': 'Tecnologia'},
            {'ticker': 'JPM', 'name': 'JPMorgan Chase & Co.', 'sector': 'Financeiro'},
            {'ticker': 'V', 'name': 'Visa Inc.', 'sector': 'Financeiro'},
            {'ticker': 'JNJ', 'name': 'Johnson & Johnson', 'sector': 'Saúde'},
            {'ticker': 'WMT', 'name': 'Walmart Inc.', 'sector': 'Varejo'},
            {'ticker': 'PG', 'name': 'Procter & Gamble Co.', 'sector': 'Consumo'},
            {'ticker': 'MA', 'name': 'Mastercard Inc.', 'sector': 'Financeiro'},
            {'ticker': 'UNH', 'name': 'UnitedHealth Group Inc.', 'sector': 'Saúde'},
            {'ticker': 'HD', 'name': 'Home Depot Inc.', 'sector': 'Varejo'}
        ]
        
        # Combina as listas
        all_stocks = br_stocks + us_stocks
        
        # Filtra os resultados
        query = query.lower()
        results = [
            stock for stock in all_stocks
            if query in stock['ticker'].lower() or query in stock['name'].lower()
        ]
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro na busca de ativos: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro na busca: {str(e)}'}), 500
