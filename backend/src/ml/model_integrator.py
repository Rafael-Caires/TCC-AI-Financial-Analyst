"""
Módulo de integração avançada de modelos de ML

Este arquivo implementa um sistema ensemble avançado para integração 
de diferentes modelos de ML para previsão de séries temporais financeiras.

Implementa:
- Ensemble com votação ponderada
- Meta-learning para otimização de pesos
- Cross-validation temporal
- Análise de incerteza
- Detecção de regime de mercado

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
from typing import Dict, List, Any, Tuple, Optional
import warnings
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from scipy import stats
import seaborn as sns

from src.ml.lstm_model import LSTMModel
from src.ml.random_forest_model import RandomForestModel
from src.ml.lightgbm_model import LightGBMModel

warnings.filterwarnings('ignore')

class AdvancedModelIntegrator:
    """
    Classe avançada para integração de modelos de ML com ensemble inteligente.
    
    Funcionalidades:
    - Ensemble adaptativo com pesos dinâmicos
    - Meta-learning para otimização automática
    - Análise de incerteza e intervalos de confiança
    - Detecção de regime de mercado
    - Cross-validation temporal
    """
    
    def __init__(self, models_dir='models'):
        """
        Inicializa o integrador avançado de modelos.
        
        Args:
            models_dir (str): Diretório para salvar/carregar modelos
        """
        self.models_dir = models_dir
        self.models = {
            'lstm': None,
            'random_forest': None,
            'lightgbm': None
        }
        
        # Pesos adaptativos iniciais
        self.static_weights = {
            'lstm': 0.4,
            'random_forest': 0.3,
            'lightgbm': 0.3
        }
        
        # Histórico de performance para pesos dinâmicos
        self.performance_history = {
            'lstm': [],
            'random_forest': [],
            'lightgbm': []
        }
        
        # Meta-learner para combinação adaptativa
        self.meta_learner = None
        self.use_meta_learning = False
        
        # Configurações de ensemble
        self.ensemble_methods = ['weighted_average', 'stacking', 'voting']
        self.current_method = 'weighted_average'
        
        # Cache para otimização
        self.prediction_cache = {}
        self.uncertainty_cache = {}
        
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
    
    def adaptive_ensemble_predict(self, data: pd.DataFrame, days_ahead: int = 30, 
                                 confidence_level: float = 0.95) -> Dict[str, Any]:
        """
        Realiza previsões usando ensemble adaptativo com análise de incerteza.
        
        Args:
            data: DataFrame com dados históricos
            days_ahead: Número de dias para prever
            confidence_level: Nível de confiança para intervalos
            
        Returns:
            Dict com previsões, incertezas e métricas de confiança
        """
        try:
            # Detecção de regime de mercado
            market_regime = self._detect_market_regime(data)
            
            # Ajuste dinâmico de pesos baseado no regime
            dynamic_weights = self._adjust_weights_by_regime(market_regime)
            
            # Coleta previsões de todos os modelos
            model_predictions = {}
            model_uncertainties = {}
            
            for name, model in self.models.items():
                if model is not None:
                    try:
                        pred = model.predict(data, days_ahead=days_ahead)
                        model_predictions[name] = pred
                        
                        # Calcula incerteza do modelo individual
                        uncertainty = self._calculate_model_uncertainty(model, data, days_ahead)
                        model_uncertainties[name] = uncertainty
                        
                    except Exception as e:
                        print(f"Erro na previsão do modelo {name}: {e}")
                        continue
            
            if not model_predictions:
                raise ValueError("Nenhum modelo disponível para previsões")
            
            # Combina previsões com pesos adaptativos
            ensemble_prediction = self._adaptive_combination(
                model_predictions, dynamic_weights, model_uncertainties
            )
            
            # Calcula intervalos de confiança ensemble
            confidence_intervals = self._calculate_ensemble_confidence_intervals(
                model_predictions, model_uncertainties, confidence_level
            )
            
            # Análise de consenso entre modelos
            consensus_analysis = self._analyze_model_consensus(model_predictions)
            
            # Score de confiança geral
            confidence_score = self._calculate_confidence_score(
                consensus_analysis, model_uncertainties, market_regime
            )
            
            return {
                'predictions': ensemble_prediction,
                'confidence_intervals': confidence_intervals,
                'market_regime': market_regime,
                'dynamic_weights': dynamic_weights,
                'model_consensus': consensus_analysis,
                'confidence_score': confidence_score,
                'individual_predictions': model_predictions,
                'uncertainty_analysis': model_uncertainties,
                'ensemble_method': self.current_method,
                'generation_timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            print(f"Erro no ensemble adaptativo: {e}")
            return self._fallback_prediction(data, days_ahead)
    
    def _detect_market_regime(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detecta o regime atual do mercado (alta volatilidade, tendência, etc.).
        
        Args:
            data: DataFrame com dados históricos
            
        Returns:
            Dict com informações do regime detectado
        """
        try:
            prices = data['Close'].values if 'Close' in data.columns else data.iloc[:, 0].values
            returns = np.diff(np.log(prices))
            
            # Volatilidade realizada
            vol_window = min(30, len(returns) // 4)
            current_vol = np.std(returns[-vol_window:]) * np.sqrt(252)
            historical_vol = np.std(returns) * np.sqrt(252)
            
            # Tendência
            trend_window = min(20, len(prices) // 5)
            recent_trend = (prices[-1] / prices[-trend_window] - 1) if len(prices) >= trend_window else 0
            
            # Momentum
            momentum_short = (prices[-1] / prices[-5] - 1) if len(prices) >= 5 else 0
            momentum_long = (prices[-1] / prices[-20] - 1) if len(prices) >= 20 else 0
            
            # Classificação do regime
            if current_vol > historical_vol * 1.5:
                volatility_regime = 'high'
            elif current_vol < historical_vol * 0.7:
                volatility_regime = 'low'
            else:
                volatility_regime = 'normal'
            
            if recent_trend > 0.1:
                trend_regime = 'strong_uptrend'
            elif recent_trend > 0.05:
                trend_regime = 'uptrend'
            elif recent_trend < -0.1:
                trend_regime = 'strong_downtrend'
            elif recent_trend < -0.05:
                trend_regime = 'downtrend'
            else:
                trend_regime = 'sideways'
            
            return {
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'current_volatility': current_vol,
                'historical_volatility': historical_vol,
                'recent_trend': recent_trend,
                'momentum_short': momentum_short,
                'momentum_long': momentum_long,
                'regime_confidence': min(1.0, abs(current_vol - historical_vol) / historical_vol)
            }
            
        except Exception as e:
            print(f"Erro na detecção de regime: {e}")
            return {
                'volatility_regime': 'unknown',
                'trend_regime': 'unknown',
                'regime_confidence': 0.0
            }
    
    def _adjust_weights_by_regime(self, market_regime: Dict[str, Any]) -> Dict[str, float]:
        """
        Ajusta pesos dos modelos baseado no regime de mercado detectado.
        
        Args:
            market_regime: Informações do regime de mercado
            
        Returns:
            Dict com pesos ajustados
        """
        base_weights = self.static_weights.copy()
        
        volatility_regime = market_regime.get('volatility_regime', 'normal')
        trend_regime = market_regime.get('trend_regime', 'sideways')
        
        # Ajustes baseados na volatilidade
        if volatility_regime == 'high':
            # Em alta volatilidade, LSTM geralmente performa melhor
            base_weights['lstm'] *= 1.3
            base_weights['random_forest'] *= 0.8
            base_weights['lightgbm'] *= 0.9
        elif volatility_regime == 'low':
            # Em baixa volatilidade, modelos tree-based podem ser melhores
            base_weights['lstm'] *= 0.8
            base_weights['random_forest'] *= 1.2
            base_weights['lightgbm'] *= 1.1
        
        # Ajustes baseados na tendência
        if 'strong' in trend_regime:
            # Em tendências fortes, LSTM captura melhor a sequência
            base_weights['lstm'] *= 1.2
        elif trend_regime == 'sideways':
            # Em mercados laterais, ensemble equilibrado
            pass  # Mantém pesos base
        
        # Normaliza os pesos
        total_weight = sum(base_weights.values())
        normalized_weights = {k: v / total_weight for k, v in base_weights.items()}
        
        return normalized_weights
    
    def _calculate_model_uncertainty(self, model: Any, data: pd.DataFrame, 
                                   days_ahead: int) -> Dict[str, float]:
        """
        Calcula medidas de incerteza para um modelo específico.
        
        Args:
            model: Modelo treinado
            data: Dados históricos
            days_ahead: Horizonte de previsão
            
        Returns:
            Dict com métricas de incerteza
        """
        try:
            # Simulação bootstrap para estimar incerteza
            n_bootstrap = 20
            predictions_bootstrap = []
            
            for _ in range(n_bootstrap):
                # Amostra bootstrap dos dados
                sample_size = min(len(data), 100)
                bootstrap_indices = np.random.choice(len(data), sample_size, replace=True)
                bootstrap_data = data.iloc[bootstrap_indices]
                
                try:
                    pred = model.predict(bootstrap_data, days_ahead=min(days_ahead, 7))
                    if pred and 'predictions' in pred:
                        predictions_bootstrap.append([p['predicted_price'] for p in pred['predictions']])
                except:
                    continue
            
            if predictions_bootstrap:
                predictions_array = np.array(predictions_bootstrap)
                
                # Calcula estatísticas de incerteza
                prediction_std = np.std(predictions_array, axis=0).mean()
                prediction_range = np.ptp(predictions_array, axis=0).mean()
                coefficient_variation = prediction_std / (np.mean(predictions_array) + 1e-8)
                
                return {
                    'standard_deviation': float(prediction_std),
                    'prediction_range': float(prediction_range),
                    'coefficient_variation': float(coefficient_variation),
                    'confidence_level': max(0, min(1, 1 - coefficient_variation))
                }
            
        except Exception as e:
            print(f"Erro no cálculo de incerteza: {e}")
        
        # Retorno padrão se houver erro
        return {
            'standard_deviation': 0.1,
            'prediction_range': 0.2,
            'coefficient_variation': 0.15,
            'confidence_level': 0.7
        }
    
    def _adaptive_combination(self, model_predictions: Dict[str, Any], 
                            weights: Dict[str, float],
                            uncertainties: Dict[str, Dict[str, float]]) -> List[Dict[str, Any]]:
        """
        Combina previsões de modelos usando pesos adaptativos e incertezas.
        
        Args:
            model_predictions: Previsões de cada modelo
            weights: Pesos dos modelos
            uncertainties: Incertezas de cada modelo
            
        Returns:
            Lista de previsões combinadas
        """
        if not model_predictions:
            return []
        
        # Pega o número de dias da primeira previsão
        first_pred = next(iter(model_predictions.values()))
        n_days = len(first_pred.get('predictions', []))
        
        combined_predictions = []
        
        for day_idx in range(n_days):
            day_predictions = []
            day_weights = []
            
            for model_name, prediction in model_predictions.items():
                if day_idx < len(prediction['predictions']):
                    pred_value = prediction['predictions'][day_idx]['predicted_price']
                    
                    # Ajusta peso baseado na incerteza
                    uncertainty_factor = uncertainties.get(model_name, {}).get('confidence_level', 0.7)
                    adjusted_weight = weights.get(model_name, 0) * uncertainty_factor
                    
                    day_predictions.append(pred_value)
                    day_weights.append(adjusted_weight)
            
            if day_predictions and sum(day_weights) > 0:
                # Normaliza pesos
                total_weight = sum(day_weights)
                normalized_weights = [w / total_weight for w in day_weights]
                
                # Calcula previsão ponderada
                combined_price = sum(p * w for p, w in zip(day_predictions, normalized_weights))
                
                # Estima intervalos baseado na dispersão
                price_std = np.std(day_predictions) if len(day_predictions) > 1 else combined_price * 0.05
                
                # Data da previsão
                pred_date = first_pred['predictions'][day_idx]['date']
                
                combined_predictions.append({
                    'date': pred_date,
                    'predicted_price': float(combined_price),
                    'lower_bound': float(combined_price - 1.96 * price_std),
                    'upper_bound': float(combined_price + 1.96 * price_std),
                    'prediction_std': float(price_std),
                    'model_agreement': float(1 - (price_std / combined_price) if combined_price != 0 else 0)
                })
        
        return combined_predictions
    
    def _calculate_ensemble_confidence_intervals(self, model_predictions: Dict[str, Any],
                                               uncertainties: Dict[str, Dict[str, float]],
                                               confidence_level: float) -> Dict[str, Any]:
        """
        Calcula intervalos de confiança para o ensemble.
        
        Args:
            model_predictions: Previsões dos modelos
            uncertainties: Incertezas dos modelos
            confidence_level: Nível de confiança desejado
            
        Returns:
            Dict com intervalos de confiança
        """
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        # Calcula incerteza média do ensemble
        avg_uncertainty = np.mean([u.get('standard_deviation', 0.1) for u in uncertainties.values()])
        
        # Dispersão entre modelos
        if len(model_predictions) > 1:
            all_predictions = []
            for pred in model_predictions.values():
                if 'predictions' in pred:
                    prices = [p['predicted_price'] for p in pred['predictions']]
                    all_predictions.append(prices)
            
            if all_predictions:
                ensemble_std = np.std(all_predictions, axis=0).mean()
            else:
                ensemble_std = avg_uncertainty
        else:
            ensemble_std = avg_uncertainty
        
        # Combina incertezas
        total_uncertainty = np.sqrt(avg_uncertainty**2 + ensemble_std**2)
        
        return {
            'confidence_level': confidence_level,
            'uncertainty_estimate': float(total_uncertainty),
            'interval_width': float(2 * z_score * total_uncertainty),
            'z_score': float(z_score),
            'individual_uncertainty': float(avg_uncertainty),
            'ensemble_uncertainty': float(ensemble_std)
        }
    
    def _analyze_model_consensus(self, model_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa o consenso entre diferentes modelos.
        
        Args:
            model_predictions: Previsões dos modelos
            
        Returns:
            Dict com análise de consenso
        """
        if len(model_predictions) < 2:
            return {'consensus_score': 1.0, 'agreement_level': 'single_model'}
        
        # Extrai previsões de preços
        price_predictions = {}
        for name, pred in model_predictions.items():
            if 'predictions' in pred:
                prices = [p['predicted_price'] for p in pred['predictions']]
                price_predictions[name] = prices
        
        if len(price_predictions) < 2:
            return {'consensus_score': 1.0, 'agreement_level': 'insufficient_data'}
        
        # Calcula correlações entre modelos
        correlations = []
        models = list(price_predictions.keys())
        
        for i in range(len(models)):
            for j in range(i + 1, len(models)):
                try:
                    corr = np.corrcoef(price_predictions[models[i]], price_predictions[models[j]])[0, 1]
                    if not np.isnan(corr):
                        correlations.append(corr)
                except:
                    continue
        
        # Calcula consenso baseado na dispersão
        all_prices = list(price_predictions.values())
        day_dispersions = []
        
        for day in range(len(all_prices[0])):
            day_prices = [prices[day] for prices in all_prices if day < len(prices)]
            if len(day_prices) > 1:
                day_std = np.std(day_prices)
                day_mean = np.mean(day_prices)
                cv = day_std / day_mean if day_mean != 0 else 1.0
                day_dispersions.append(cv)
        
        avg_correlation = np.mean(correlations) if correlations else 0.0
        avg_dispersion = np.mean(day_dispersions) if day_dispersions else 1.0
        
        # Score de consenso (combina correlação e baixa dispersão)
        consensus_score = (avg_correlation + (1 - min(avg_dispersion, 1.0))) / 2
        
        # Classifica nível de consenso
        if consensus_score > 0.8:
            agreement_level = 'high'
        elif consensus_score > 0.6:
            agreement_level = 'moderate'
        elif consensus_score > 0.4:
            agreement_level = 'low'
        else:
            agreement_level = 'poor'
        
        return {
            'consensus_score': float(consensus_score),
            'agreement_level': agreement_level,
            'average_correlation': float(avg_correlation),
            'average_dispersion': float(avg_dispersion),
            'model_count': len(price_predictions)
        }
    
    def _calculate_confidence_score(self, consensus: Dict[str, Any], 
                                  uncertainties: Dict[str, Dict[str, float]],
                                  market_regime: Dict[str, Any]) -> float:
        """
        Calcula score geral de confiança das previsões.
        
        Args:
            consensus: Análise de consenso entre modelos
            uncertainties: Incertezas individuais
            market_regime: Regime de mercado detectado
            
        Returns:
            Score de confiança entre 0 e 1
        """
        # Score base do consenso
        consensus_score = consensus.get('consensus_score', 0.5)
        
        # Score médio de confiança dos modelos individuais
        individual_confidence = np.mean([
            u.get('confidence_level', 0.5) for u in uncertainties.values()
        ])
        
        # Penalização por regime de mercado incerto
        regime_confidence = market_regime.get('regime_confidence', 0.5)
        
        # Score combinado
        combined_score = (
            consensus_score * 0.4 +
            individual_confidence * 0.4 +
            regime_confidence * 0.2
        )
        
        return max(0.0, min(1.0, combined_score))
    
    def _fallback_prediction(self, data: pd.DataFrame, days_ahead: int) -> Dict[str, Any]:
        """
        Previsão de fallback em caso de erro no ensemble.
        
        Args:
            data: Dados históricos
            days_ahead: Dias para prever
            
        Returns:
            Previsão básica de fallback
        """
        try:
            # Usa último preço como baseline
            last_price = data['Close'].iloc[-1] if 'Close' in data.columns else data.iloc[-1, 0]
            
            # Estima volatilidade simples
            returns = data['Close'].pct_change().dropna() if 'Close' in data.columns else data.iloc[:, 0].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)
            
            predictions = []
            for i in range(days_ahead):
                date = (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d')
                
                predictions.append({
                    'date': date,
                    'predicted_price': float(last_price),
                    'lower_bound': float(last_price * (1 - volatility * 0.1)),
                    'upper_bound': float(last_price * (1 + volatility * 0.1)),
                    'confidence': 0.3  # Baixa confiança para fallback
                })
            
            return {
                'predictions': predictions,
                'method': 'fallback',
                'confidence_score': 0.3,
                'warning': 'Usando previsão de fallback devido a erro no ensemble'
            }
            
        except Exception as e:
            print(f"Erro na previsão de fallback: {e}")
            return {'error': 'Não foi possível gerar previsões'}

# Mantém compatibilidade com código existente
ModelIntegrator = AdvancedModelIntegrator
