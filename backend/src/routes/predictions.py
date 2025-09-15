"""
Rotas para previsões de séries temporais financeiras

Este arquivo implementa as rotas para previsões de séries temporais financeiras
usando modelos de machine learning.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

from flask import Blueprint, request, jsonify, current_app
from src.services.financial_data_service import FinancialDataService
from src.ml.model_integrator import ModelIntegrator
from src.utils.auth import token_required
import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt
import io
import base64
from datetime import datetime, timedelta

# Cria o blueprint para as rotas de previsões
predictions_bp = Blueprint('predictions', __name__, url_prefix='/api/predictions')

# Inicializa o integrador de modelos
model_integrator = ModelIntegrator(models_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'models'))

# Rota específica para previsões por ticker (compatível com frontend)
@predictions_bp.route('/<ticker>', methods=['GET'])
def get_prediction_by_ticker(ticker):
    """
    Obtém previsão para um ativo específico via URL.
    
    Args:
        ticker (str): Ticker do ativo
        
    Returns:
        JSON: Previsão para o ativo
    """
    try:
        days = int(request.args.get('days', 7))
        
        # Dados simulados de previsão para desenvolvimento
        prediction_data = {
            'success': True,
            'ticker': ticker,
            'current_price': 32.45 if ticker == 'PETR4' else 25.80,
            'prediction_days': days,
            'predictions': [
                {
                    'date': (datetime.now() + timedelta(days=i)).strftime('%Y-%m-%d'),
                    'predicted_price': round(32.45 + (i * 0.5) + np.random.normal(0, 0.3), 2),
                    'confidence': round((0.85 - (i * 0.02)) * 100, 1),  # Convertendo para porcentagem e arredondando
                    'change_percentage': round(np.random.normal(2.0, 1.5), 1),  # Variação percentual simulada
                    'trend': 'up' if np.random.random() > 0.4 else 'down'  # Trend aleatório com viés positivo
                } for i in range(1, days + 1)
            ],
            'analysis': {
                'trend': 'bullish',
                'volatility': 'medium',
                'support_level': 30.50,
                'resistance_level': 35.20,
                'recommendation': 'BUY'
            }
        }
        
        return jsonify(prediction_data)
        
    except Exception as e:
        current_app.logger.error(f"Erro na previsão: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Erro ao obter previsão: {str(e)}'
        }), 500

# Rota para obter previsão para um ativo
@predictions_bp.route('/forecast', methods=['GET'])
@token_required
def get_forecast(user_id):
    """
    Obtém previsão para um ativo.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Previsão para o ativo
    """
    # Obtém parâmetros da requisição
    ticker = request.args.get('ticker')
    days_ahead = int(request.args.get('days_ahead', 30))
    model_type = request.args.get('model_type', 'ensemble')  # 'lstm', 'random_forest', 'lightgbm', 'ensemble'
    
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker não fornecido'}), 400
    
    try:
        # Obtém dados históricos (2 anos para treinamento)
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        data = FinancialDataService.get_stock_data(ticker, start_date=start_date)
        
        # Verifica se há dados suficientes
        if len(data) < 252:  # Pelo menos 1 ano de dados
            return jsonify({
                'success': False, 
                'message': 'Não há dados históricos suficientes para fazer previsões'
            }), 400
        
        # Carrega os modelos
        try:
            model_integrator.load_all_models()
            current_app.logger.info("Modelos carregados com sucesso")
        except Exception as e:
            current_app.logger.warning(f"Erro ao carregar modelos: {str(e)}. Treinando novos modelos...")
            # Se não conseguir carregar, treina novos modelos
            model_integrator.train_all_models(data, verbose=0)
            model_integrator.save_all_models()
        
        # Faz a previsão
        if model_type == 'ensemble':
            prediction = model_integrator.predict(data, days_ahead=days_ahead, use_ensemble=True)
        elif model_type in model_integrator.models and model_integrator.models[model_type] is not None:
            prediction = model_integrator.models[model_type].predict(data, days_ahead=days_ahead)
        else:
            return jsonify({
                'success': False, 
                'message': f'Modelo {model_type} não disponível'
            }), 400
        
        # Gera gráfico da previsão
        if model_type == 'ensemble':
            fig = model_integrator.plot_combined_predictions(data, prediction)
        else:
            fig = model_integrator.models[model_type].plot_predictions(data, prediction)
        
        # Converte o gráfico para base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Adiciona o gráfico à resposta
        prediction['chart'] = chart_base64
        
        return jsonify({
            'success': True,
            'model_type': model_type,
            'prediction': prediction
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao gerar previsão: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao gerar previsão: {str(e)}'}), 500

# Rota para comparar modelos
@predictions_bp.route('/compare-models', methods=['GET'])
@token_required
def compare_models(user_id):
    """
    Compara diferentes modelos de previsão para um ativo.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Comparação de modelos
    """
    # Obtém parâmetros da requisição
    ticker = request.args.get('ticker')
    days_ahead = int(request.args.get('days_ahead', 30))
    
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker não fornecido'}), 400
    
    try:
        # Obtém dados históricos (2 anos para treinamento)
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        data = FinancialDataService.get_stock_data(ticker, start_date=start_date)
        
        # Verifica se há dados suficientes
        if len(data) < 252:  # Pelo menos 1 ano de dados
            return jsonify({
                'success': False, 
                'message': 'Não há dados históricos suficientes para fazer previsões'
            }), 400
        
        # Carrega os modelos
        try:
            model_integrator.load_all_models()
        except Exception as e:
            current_app.logger.warning(f"Erro ao carregar modelos: {str(e)}. Treinando novos modelos...")
            # Se não conseguir carregar, treina novos modelos
            model_integrator.train_all_models(data, verbose=0)
            model_integrator.save_all_models()
        
        # Faz previsões com cada modelo
        predictions = {}
        for model_name, model in model_integrator.models.items():
            if model is not None:
                try:
                    predictions[model_name] = model.predict(data, days_ahead=days_ahead)
                except Exception as e:
                    current_app.logger.error(f"Erro ao fazer previsão com modelo {model_name}: {str(e)}")
        
        # Faz previsão com o ensemble
        predictions['ensemble'] = model_integrator.predict(data, days_ahead=days_ahead, use_ensemble=True)
        
        # Gera gráfico de comparação
        fig = model_integrator.plot_model_comparison(data, days_to_plot=days_ahead)
        
        # Converte o gráfico para base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        chart_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        # Prepara a resposta
        comparison = {
            'ticker': ticker,
            'days_ahead': days_ahead,
            'chart': chart_base64,
            'predictions': {}
        }
        
        # Adiciona métricas de cada modelo
        for model_name, prediction in predictions.items():
            comparison['predictions'][model_name] = {
                'last_price': prediction['last_price'],
                'final_price': prediction['predictions'][-1]['predicted_price'],
                'change_percent': ((prediction['predictions'][-1]['predicted_price'] / prediction['last_price']) - 1) * 100,
                'confidence_interval': [
                    prediction['predictions'][-1]['lower_bound'],
                    prediction['predictions'][-1]['upper_bound']
                ]
            }
        
        return jsonify({
            'success': True,
            'comparison': comparison
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao comparar modelos: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao comparar modelos: {str(e)}'}), 500

# Rota para avaliar modelos
@predictions_bp.route('/evaluate-models', methods=['GET'])
@token_required
def evaluate_models(user_id):
    """
    Avalia o desempenho dos modelos em dados históricos.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Avaliação dos modelos
    """
    # Obtém parâmetros da requisição
    ticker = request.args.get('ticker')
    
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker não fornecido'}), 400
    
    try:
        # Obtém dados históricos (3 anos para ter dados suficientes para treino e teste)
        start_date = (datetime.now() - timedelta(days=1095)).strftime('%Y-%m-%d')
        data = FinancialDataService.get_stock_data(ticker, start_date=start_date)
        
        # Verifica se há dados suficientes
        if len(data) < 504:  # Pelo menos 2 anos de dados
            return jsonify({
                'success': False, 
                'message': 'Não há dados históricos suficientes para avaliar modelos'
            }), 400
        
        # Divide os dados em treino e teste
        train_size = int(len(data) * 0.8)
        train_data = data.iloc[:train_size]
        test_data = data.iloc[train_size:]
        
        # Carrega ou treina os modelos
        try:
            model_integrator.load_all_models()
        except Exception as e:
            current_app.logger.warning(f"Erro ao carregar modelos: {str(e)}. Treinando novos modelos...")
            # Se não conseguir carregar, treina novos modelos
            model_integrator.train_all_models(train_data, verbose=0)
            model_integrator.save_all_models()
        
        # Avalia os modelos
        evaluation = model_integrator.evaluate_all_models(test_data)
        
        # Prepara a resposta
        result = {
            'ticker': ticker,
            'train_period': {
                'start': train_data.index[0].strftime('%Y-%m-%d'),
                'end': train_data.index[-1].strftime('%Y-%m-%d'),
                'days': len(train_data)
            },
            'test_period': {
                'start': test_data.index[0].strftime('%Y-%m-%d'),
                'end': test_data.index[-1].strftime('%Y-%m-%d'),
                'days': len(test_data)
            },
            'evaluation': evaluation
        }
        
        return jsonify({
            'success': True,
            'result': result
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao avaliar modelos: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao avaliar modelos: {str(e)}'}), 500

# Rota para treinar modelos
@predictions_bp.route('/train-models', methods=['POST'])
@token_required
def train_models(user_id):
    """
    Treina os modelos com dados históricos.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Status do treinamento
    """
    data = request.get_json()
    
    # Obtém parâmetros
    ticker = data.get('ticker')
    
    if not ticker:
        return jsonify({'success': False, 'message': 'Ticker não fornecido'}), 400
    
    try:
        # Obtém dados históricos (2 anos para treinamento)
        start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
        stock_data = FinancialDataService.get_stock_data(ticker, start_date=start_date)
        
        # Verifica se há dados suficientes
        if len(stock_data) < 252:  # Pelo menos 1 ano de dados
            return jsonify({
                'success': False, 
                'message': 'Não há dados históricos suficientes para treinar modelos'
            }), 400
        
        # Treina os modelos
        model_integrator.train_all_models(stock_data, verbose=0)
        
        # Salva os modelos
        paths = model_integrator.save_all_models()
        
        return jsonify({
            'success': True,
            'message': 'Modelos treinados com sucesso',
            'models': list(paths.keys())
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao treinar modelos: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao treinar modelos: {str(e)}'}), 500

# Rota para obter previsões para múltiplos ativos
@predictions_bp.route('/batch-forecast', methods=['POST'])
@token_required
def batch_forecast(user_id):
    """
    Obtém previsões para múltiplos ativos.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Previsões para os ativos
    """
    data = request.get_json()
    
    # Obtém parâmetros
    tickers = data.get('tickers', [])
    days_ahead = data.get('days_ahead', 30)
    model_type = data.get('model_type', 'ensemble')
    
    if not tickers:
        return jsonify({'success': False, 'message': 'Nenhum ticker fornecido'}), 400
    
    try:
        # Carrega os modelos
        try:
            model_integrator.load_all_models()
        except Exception as e:
            current_app.logger.warning(f"Erro ao carregar modelos: {str(e)}. Os modelos serão treinados para cada ativo.")
        
        # Faz previsões para cada ativo
        results = {}
        
        for ticker in tickers:
            try:
                # Obtém dados históricos
                start_date = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
                data = FinancialDataService.get_stock_data(ticker, start_date=start_date)
                
                # Verifica se há dados suficientes
                if len(data) < 252:  # Pelo menos 1 ano de dados
                    results[ticker] = {
                        'success': False,
                        'message': 'Dados históricos insuficientes'
                    }
                    continue
                
                # Treina modelos se necessário
                if not all(model is not None for model in model_integrator.models.values()):
                    model_integrator.train_all_models(data, verbose=0)
                    model_integrator.save_all_models()
                
                # Faz a previsão
                if model_type == 'ensemble':
                    prediction = model_integrator.predict(data, days_ahead=days_ahead, use_ensemble=True)
                elif model_type in model_integrator.models and model_integrator.models[model_type] is not None:
                    prediction = model_integrator.models[model_type].predict(data, days_ahead=days_ahead)
                else:
                    results[ticker] = {
                        'success': False,
                        'message': f'Modelo {model_type} não disponível'
                    }
                    continue
                
                # Adiciona à resposta
                results[ticker] = {
                    'success': True,
                    'prediction': prediction
                }
            
            except Exception as e:
                current_app.logger.error(f"Erro ao gerar previsão para {ticker}: {str(e)}")
                results[ticker] = {
                    'success': False,
                    'message': f'Erro: {str(e)}'
                }
        
        return jsonify({
            'success': True,
            'results': results
        }), 200
    
    except Exception as e:
        current_app.logger.error(f"Erro ao processar lote de previsões: {str(e)}")
        return jsonify({'success': False, 'message': f'Erro ao processar lote: {str(e)}'}), 500
