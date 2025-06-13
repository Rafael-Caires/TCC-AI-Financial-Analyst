"""
Módulo de integração de modelos de ML

Este arquivo implementa a integração dos diferentes modelos de ML
para previsão de séries temporais financeiras.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib
import json

from src.ml.lstm_model import LSTMModel
from src.ml.random_forest_model import RandomForestModel
from src.ml.lightgbm_model import LightGBMModel

class ModelIntegrator:
    """
    Classe para integrar diferentes modelos de ML e gerar previsões combinadas.
    """
    
    def __init__(self, models_dir='models'):
        """
        Inicializa o integrador de modelos.
        
        Args:
            models_dir (str): Diretório para salvar/carregar modelos
        """
        self.models_dir = models_dir
        self.models = {
            'lstm': None,
            'random_forest': None,
            'lightgbm': None
        }
        self.weights = {
            'lstm': 0.4,
            'random_forest': 0.3,
            'lightgbm': 0.3
        }
        
        # Cria o diretório de modelos se não existir
        os.makedirs(models_dir, exist_ok=True)
    
    def train_all_models(self, data, test_size=0.2, verbose=1):
        """
        Treina todos os modelos com os dados fornecidos.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados de preços
            test_size (float): Proporção dos dados para teste
            verbose (int): Nível de verbosidade durante o treinamento
            
        Returns:
            self: Instância com os modelos treinados
        """
        # Treina o modelo LSTM
        if verbose:
            print("Treinando modelo LSTM...")
        
        lstm_model = LSTMModel(
            sequence_length=60,
            epochs=50,
            batch_size=32,
            model_path=os.path.join(self.models_dir, 'lstm_model')
        )
        self.models['lstm'] = lstm_model.train(data, validation_split=test_size, verbose=verbose)
        
        # Treina o modelo Random Forest
        if verbose:
            print("\nTreinando modelo Random Forest...")
        
        rf_model = RandomForestModel(
            n_estimators=100,
            max_depth=None,
            sequence_length=10,
            model_path=os.path.join(self.models_dir, 'rf_model')
        )
        self.models['random_forest'] = rf_model.train(data, test_size=test_size, verbose=verbose)
        
        # Treina o modelo LightGBM
        if verbose:
            print("\nTreinando modelo LightGBM...")
        
        lgb_model = LightGBMModel(
            num_leaves=31,
            learning_rate=0.05,
            n_estimators=100,
            sequence_length=10,
            model_path=os.path.join(self.models_dir, 'lgb_model')
        )
        self.models['lightgbm'] = lgb_model.train(data, test_size=test_size, verbose=verbose)
        
        return self
    
    def save_all_models(self):
        """
        Salva todos os modelos treinados.
        
        Returns:
            dict: Dicionário com os caminhos onde os modelos foram salvos
        """
        paths = {}
        
        for name, model in self.models.items():
            if model is not None:
                path = model.save(os.path.join(self.models_dir, f'{name}_model'))
                paths[name] = path
        
        # Salva os pesos
        with open(os.path.join(self.models_dir, 'weights.json'), 'w') as f:
            json.dump(self.weights, f)
        
        return paths
    
    def load_all_models(self):
        """
        Carrega todos os modelos salvos.
        
        Returns:
            self: Instância com os modelos carregados
        """
        # Carrega o modelo LSTM
        try:
            lstm_model = LSTMModel()
            self.models['lstm'] = lstm_model.load(os.path.join(self.models_dir, 'lstm_model'))
        except Exception as e:
            print(f"Erro ao carregar modelo LSTM: {e}")
        
        # Carrega o modelo Random Forest
        try:
            rf_model = RandomForestModel()
            self.models['random_forest'] = rf_model.load(os.path.join(self.models_dir, 'rf_model'))
        except Exception as e:
            print(f"Erro ao carregar modelo Random Forest: {e}")
        
        # Carrega o modelo LightGBM
        try:
            lgb_model = LightGBMModel()
            self.models['lightgbm'] = lgb_model.load(os.path.join(self.models_dir, 'lgb_model'))
        except Exception as e:
            print(f"Erro ao carregar modelo LightGBM: {e}")
        
        # Carrega os pesos
        try:
            with open(os.path.join(self.models_dir, 'weights.json'), 'r') as f:
                self.weights = json.load(f)
        except Exception as e:
            print(f"Erro ao carregar pesos: {e}")
        
        return self
    
    def predict(self, data, days_ahead=30, use_ensemble=True):
        """
        Faz previsões usando os modelos treinados.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados históricos
            days_ahead (int): Número de dias para prever à frente
            use_ensemble (bool): Se True, combina as previsões dos modelos
            
        Returns:
            dict: Dicionário com previsões, datas e intervalos de confiança
        """
        predictions = {}
        
        # Verifica se há modelos carregados
        if all(model is None for model in self.models.values()):
            raise ValueError("Nenhum modelo carregado. Treine ou carregue os modelos primeiro.")
        
        # Faz previsões com cada modelo disponível
        for name, model in self.models.items():
            if model is not None:
                try:
                    predictions[name] = model.predict(data, days_ahead=days_ahead)
                except Exception as e:
                    print(f"Erro ao fazer previsões com o modelo {name}: {e}")
        
        # Se não há previsões, retorna erro
        if not predictions:
            raise ValueError("Não foi possível fazer previsões com nenhum modelo.")
        
        # Se não for para usar ensemble ou só há um modelo, retorna a previsão do primeiro modelo
        if not use_ensemble or len(predictions) == 1:
            return next(iter(predictions.values()))
        
        # Combina as previsões dos modelos
        return self._combine_predictions(predictions, days_ahead)
    
    def _combine_predictions(self, predictions, days_ahead):
        """
        Combina as previsões dos diferentes modelos.
        
        Args:
            predictions (dict): Dicionário com previsões de cada modelo
            days_ahead (int): Número de dias previstos
            
        Returns:
            dict: Dicionário com previsões combinadas
        """
        # Inicializa listas para armazenar resultados combinados
        combined_prices = []
        combined_lower_bounds = []
        combined_upper_bounds = []
        dates = []
        
        # Para cada dia previsto
        for day in range(days_ahead):
            day_prices = []
            day_lower_bounds = []
            day_upper_bounds = []
            
            # Coleta previsões de cada modelo para este dia
            for name, model_predictions in predictions.items():
                if name in self.weights and day < len(model_predictions['predictions']):
                    pred = model_predictions['predictions'][day]
                    day_prices.append(pred['predicted_price'] * self.weights[name])
                    day_lower_bounds.append(pred['lower_bound'] * self.weights[name])
                    day_upper_bounds.append(pred['upper_bound'] * self.weights[name])
                    
                    # Coleta a data apenas uma vez
                    if not dates and day == 0:
                        dates = [pred['date'] for pred in model_predictions['predictions']]
            
            # Calcula a média ponderada
            if day_prices:
                combined_prices.append(sum(day_prices))
                combined_lower_bounds.append(sum(day_lower_bounds))
                combined_upper_bounds.append(sum(day_upper_bounds))
        
        # Formata os resultados
        results = []
        for i in range(len(combined_prices)):
            results.append({
                'date': dates[i],
                'predicted_price': float(combined_prices[i]),
                'lower_bound': float(combined_lower_bounds[i]),
                'upper_bound': float(combined_upper_bounds[i]),
            })
        
        # Obtém o ticker e último preço do primeiro modelo
        first_model = next(iter(predictions.values()))
        
        return {
            'ticker': first_model['ticker'],
            'last_price': first_model['last_price'],
            'forecast_days': days_ahead,
            'predictions': results,
            'ensemble': True,
            'models_used': list(predictions.keys())
        }
    
    def evaluate_all_models(self, test_data):
        """
        Avalia todos os modelos em dados de teste.
        
        Args:
            test_data (pandas.DataFrame): DataFrame com dados de teste
            
        Returns:
            dict: Dicionário com métricas de avaliação para cada modelo
        """
        results = {}
        
        for name, model in self.models.items():
            if model is not None:
                try:
                    results[name] = model.evaluate(test_data)
                except Exception as e:
                    print(f"Erro ao avaliar o modelo {name}: {e}")
        
        return results
    
    def optimize_weights(self, validation_data, metric='direction_accuracy'):
        """
        Otimiza os pesos dos modelos com base em dados de validação.
        
        Args:
            validation_data (pandas.DataFrame): DataFrame com dados de validação
            metric (str): Métrica a ser otimizada ('mse', 'rmse', 'mae', 'direction_accuracy')
            
        Returns:
            dict: Dicionário com os pesos otimizados
        """
        # Avalia cada modelo
        evaluations = self.evaluate_all_models(validation_data)
        
        # Se não há avaliações, retorna os pesos padrão
        if not evaluations:
            return self.weights
        
        # Calcula os pesos com base na métrica escolhida
        total = 0
        weights = {}
        
        for name, eval_results in evaluations.items():
            if metric in eval_results:
                # Para métricas de erro (mse, rmse, mae), quanto menor, melhor
                if metric in ['mse', 'rmse', 'mae', 'mape']:
                    # Inverte para que menor erro tenha maior peso
                    value = 1 / (eval_results[metric] + 1e-10)
                else:
                    # Para acurácia, quanto maior, melhor
                    value = eval_results[metric]
                
                weights[name] = value
                total += value
        
        # Normaliza os pesos
        if total > 0:
            for name in weights:
                weights[name] /= total
        
        # Atualiza os pesos
        self.weights = weights
        
        return weights
    
    def plot_combined_predictions(self, data, predictions):
        """
        Plota os dados históricos e as previsões combinadas.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados históricos
            predictions (dict): Dicionário com previsões retornado pelo método predict
            
        Returns:
            matplotlib.figure.Figure: Figura com o gráfico das previsões
        """
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plota os dados históricos
        ax.plot(data.index[-90:], data['Close'].values[-90:], label='Dados Históricos', color='blue')
        
        # Prepara dados de previsão
        dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in predictions['predictions']]
        predicted_prices = [p['predicted_price'] for p in predictions['predictions']]
        lower_bounds = [p['lower_bound'] for p in predictions['predictions']]
        upper_bounds = [p['upper_bound'] for p in predictions['predictions']]
        
        # Plota as previsões
        ax.plot(dates, predicted_prices, label='Previsões (Ensemble)', color='orange', linestyle='--')
        
        # Plota o intervalo de confiança
        ax.fill_between(dates, lower_bounds, upper_bounds, color='orange', alpha=0.2, label='Intervalo de Confiança')
        
        # Configurações do gráfico
        ax.set_title(f"Previsão de Preços para {predictions['ticker']} (Ensemble)")
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço')
        ax.legend()
        ax.grid(True)
        
        # Formata o eixo x para mostrar datas
        fig.autofmt_xdate()
        
        return fig
    
    def plot_model_comparison(self, data, days_to_plot=30):
        """
        Plota uma comparação das previsões de diferentes modelos.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados históricos
            days_to_plot (int): Número de dias para plotar
            
        Returns:
            matplotlib.figure.Figure: Figura com o gráfico de comparação
        """
        # Faz previsões com cada modelo
        predictions = {}
        for name, model in self.models.items():
            if model is not None:
                try:
                    predictions[name] = model.predict(data, days_ahead=days_to_plot)
                except Exception as e:
                    print(f"Erro ao fazer previsões com o modelo {name}: {e}")
        
        # Faz previsão com o ensemble
        ensemble_pred = self.predict(data, days_ahead=days_to_plot)
        
        # Cria o gráfico
        fig, ax = plt.subplots(figsize=(14, 7))
        
        # Plota os dados históricos
        ax.plot(data.index[-90:], data['Close'].values[-90:], label='Dados Históricos', color='blue')
        
        # Cores para cada modelo
        colors = {
            'lstm': 'red',
            'random_forest': 'green',
            'lightgbm': 'purple',
            'ensemble': 'orange'
        }
        
        # Plota as previsões de cada modelo
        for name, pred in predictions.items():
            dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in pred['predictions']]
            predicted_prices = [p['predicted_price'] for p in pred['predictions']]
            ax.plot(dates, predicted_prices, label=f'Previsão ({name.upper()})', color=colors.get(name, 'gray'), linestyle='--')
        
        # Plota a previsão do ensemble
        dates = [datetime.strptime(p['date'], '%Y-%m-%d') for p in ensemble_pred['predictions']]
        predicted_prices = [p['predicted_price'] for p in ensemble_pred['predictions']]
        ax.plot(dates, predicted_prices, label='Previsão (ENSEMBLE)', color=colors['ensemble'], linestyle='-', linewidth=2)
        
        # Configurações do gráfico
        ax.set_title(f"Comparação de Modelos para {data.name if hasattr(data, 'name') else 'unknown'}")
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço')
        ax.legend()
        ax.grid(True)
        
        # Formata o eixo x para mostrar datas
        fig.autofmt_xdate()
        
        return fig
