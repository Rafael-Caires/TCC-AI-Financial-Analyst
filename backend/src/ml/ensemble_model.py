"""
Módulo de ensemble de modelos de machine learning

Este arquivo implementa um sistema de ensemble que combina múltiplos modelos
de machine learning para melhorar a precisão das previsões financeiras.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error
import lightgbm as lgb
import xgboost as xgb
import optuna
from typing import Dict, List, Tuple, Any
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from .lstm_model import LSTMModel
from .random_forest_model import RandomForestModel
from .lightgbm_model import LightGBMModel

class EnsembleModel:
    """
    Classe para ensemble de modelos de machine learning para previsão financeira.
    Combina LSTM, Random Forest, LightGBM e XGBoost usando diferentes estratégias.
    """
    
    def __init__(self, ensemble_method='weighted_average', optimize_weights=True):
        """
        Inicializa o ensemble de modelos.
        
        Args:
            ensemble_method (str): Método de ensemble ('weighted_average', 'voting', 'stacking')
            optimize_weights (bool): Se deve otimizar os pesos automaticamente
        """
        self.ensemble_method = ensemble_method
        self.optimize_weights = optimize_weights
        
        # Inicializa os modelos base
        self.models = {
            'lstm': LSTMModel(),
            'random_forest': RandomForestModel(),
            'lightgbm': LightGBMModel(),
            'xgboost': None  # Será inicializado no treinamento
        }
        
        # Pesos para ensemble ponderado
        self.weights = {
            'lstm': 0.4,
            'random_forest': 0.2,
            'lightgbm': 0.25,
            'xgboost': 0.15
        }
        
        # Métricas de performance individual
        self.individual_metrics = {}
        
        # Modelo de stacking (se usado)
        self.meta_model = None
        
        # Histórico de otimização
        self.optimization_history = []
        
    def _prepare_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepara features técnicas para os modelos tree-based.
        
        Args:
            data (pd.DataFrame): DataFrame com dados OHLCV
            
        Returns:
            pd.DataFrame: DataFrame com features técnicas
        """
        df = data.copy()
        
        # Features básicas
        df['returns'] = df['Close'].pct_change()
        df['log_returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Médias móveis
        for window in [5, 10, 20, 50]:
            df[f'sma_{window}'] = df['Close'].rolling(window=window).mean()
            df[f'ema_{window}'] = df['Close'].ewm(span=window).mean()
        
        # Indicadores técnicos
        # RSI
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # MACD
        ema_12 = df['Close'].ewm(span=12).mean()
        ema_26 = df['Close'].ewm(span=26).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        
        # Bollinger Bands
        sma_20 = df['Close'].rolling(window=20).mean()
        std_20 = df['Close'].rolling(window=20).std()
        df['bb_upper'] = sma_20 + (std_20 * 2)
        df['bb_lower'] = sma_20 - (std_20 * 2)
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
        df['bb_position'] = (df['Close'] - df['bb_lower']) / df['bb_width']
        
        # Volatilidade
        df['volatility'] = df['returns'].rolling(window=20).std()
        
        # Volume features
        if 'Volume' in df.columns:
            df['volume_sma'] = df['Volume'].rolling(window=20).mean()
            df['volume_ratio'] = df['Volume'] / df['volume_sma']
        
        # Lag features
        for lag in [1, 2, 3, 5]:
            df[f'close_lag_{lag}'] = df['Close'].shift(lag)
            df[f'returns_lag_{lag}'] = df['returns'].shift(lag)
        
        # Remove NaN values
        df = df.dropna()
        
        return df
    
    def _optimize_weights_optuna(self, X_val: np.ndarray, y_val: np.ndarray, 
                                predictions: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Otimiza os pesos do ensemble usando Optuna.
        
        Args:
            X_val (np.ndarray): Features de validação
            y_val (np.ndarray): Targets de validação
            predictions (Dict[str, np.ndarray]): Previsões de cada modelo
            
        Returns:
            Dict[str, float]: Pesos otimizados
        """
        def objective(trial):
            # Sugere pesos para cada modelo
            weights = {}
            total_weight = 0
            
            for model_name in self.models.keys():
                if model_name in predictions:
                    weight = trial.suggest_float(f'weight_{model_name}', 0.0, 1.0)
                    weights[model_name] = weight
                    total_weight += weight
            
            # Normaliza os pesos
            if total_weight > 0:
                weights = {k: v/total_weight for k, v in weights.items()}
            else:
                return float('inf')
            
            # Calcula previsão ensemble
            ensemble_pred = np.zeros_like(y_val)
            for model_name, weight in weights.items():
                if model_name in predictions:
                    ensemble_pred += weight * predictions[model_name]
            
            # Calcula RMSE
            rmse = np.sqrt(mean_squared_error(y_val, ensemble_pred))
            return rmse
        
        # Executa otimização
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100, show_progress_bar=False)
        
        # Extrai melhores pesos
        best_weights = {}
        total_weight = 0
        
        for model_name in self.models.keys():
            if f'weight_{model_name}' in study.best_params:
                weight = study.best_params[f'weight_{model_name}']
                best_weights[model_name] = weight
                total_weight += weight
        
        # Normaliza os pesos
        if total_weight > 0:
            best_weights = {k: v/total_weight for k, v in best_weights.items()}
        
        # Salva histórico de otimização
        self.optimization_history.append({
            'timestamp': datetime.now(),
            'best_value': study.best_value,
            'best_params': study.best_params,
            'n_trials': len(study.trials)
        })
        
        return best_weights
    
    def train(self, data: pd.DataFrame, validation_split: float = 0.2, 
              optimize_hyperparams: bool = True) -> 'EnsembleModel':
        """
        Treina todos os modelos do ensemble.
        
        Args:
            data (pd.DataFrame): DataFrame com dados históricos
            validation_split (float): Proporção para validação
            optimize_hyperparams (bool): Se deve otimizar hiperparâmetros
            
        Returns:
            EnsembleModel: Instância treinada
        """
        print("Iniciando treinamento do ensemble...")
        
        # Prepara features para modelos tree-based
        featured_data = self._prepare_features(data)
        
        # Divide dados
        split_idx = int(len(data) * (1 - validation_split))
        train_data = data.iloc[:split_idx]
        val_data = data.iloc[split_idx:]
        
        train_featured = featured_data.iloc[:split_idx]
        val_featured = featured_data.iloc[split_idx:]
        
        # Treina LSTM
        print("Treinando LSTM...")
        self.models['lstm'].train(train_data, validation_split=0.2)
        
        # Treina Random Forest
        print("Treinando Random Forest...")
        self.models['random_forest'].train(train_featured)
        
        # Treina LightGBM
        print("Treinando LightGBM...")
        self.models['lightgbm'].train(train_featured)
        
        # Treina XGBoost
        print("Treinando XGBoost...")
        feature_cols = [col for col in train_featured.columns if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]
        X_train = train_featured[feature_cols].values
        y_train = train_featured['Close'].values
        
        X_val = val_featured[feature_cols].values
        y_val = val_featured['Close'].values
        
        if optimize_hyperparams:
            # Otimização de hiperparâmetros para XGBoost
            def objective_xgb(trial):
                params = {
                    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                    'max_depth': trial.suggest_int('max_depth', 3, 10),
                    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                    'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                    'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
                    'random_state': 42
                }
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train)
                pred = model.predict(X_val)
                return mean_squared_error(y_val, pred)
            
            study_xgb = optuna.create_study(direction='minimize')
            study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=False)
            
            self.models['xgboost'] = xgb.XGBRegressor(**study_xgb.best_params)
        else:
            self.models['xgboost'] = xgb.XGBRegressor(
                n_estimators=500,
                max_depth=6,
                learning_rate=0.1,
                random_state=42
            )
        
        self.models['xgboost'].fit(X_train, y_train)
        
        # Avalia modelos individuais
        print("Avaliando modelos individuais...")
        predictions = {}
        
        # LSTM predictions
        lstm_pred = self.models['lstm'].predict(val_data, days_ahead=len(val_data))
        predictions['lstm'] = np.array([p['predicted_price'] for p in lstm_pred['predictions']])
        
        # Random Forest predictions
        rf_pred = self.models['random_forest'].predict(val_featured)
        predictions['random_forest'] = rf_pred['predictions']
        
        # LightGBM predictions
        lgb_pred = self.models['lightgbm'].predict(val_featured)
        predictions['lightgbm'] = lgb_pred['predictions']
        
        # XGBoost predictions
        xgb_pred = self.models['xgboost'].predict(X_val)
        predictions['xgboost'] = xgb_pred
        
        # Calcula métricas individuais
        y_true = val_data['Close'].values
        for model_name, pred in predictions.items():
            if len(pred) == len(y_true):
                mse = mean_squared_error(y_true, pred)
                mae = mean_absolute_error(y_true, pred)
                rmse = np.sqrt(mse)
                
                self.individual_metrics[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
        
        # Otimiza pesos se solicitado
        if self.optimize_weights and len(predictions) > 1:
            print("Otimizando pesos do ensemble...")
            # Ajusta tamanhos das previsões
            min_len = min(len(pred) for pred in predictions.values())
            adjusted_predictions = {k: v[:min_len] for k, v in predictions.items()}
            adjusted_y_val = y_true[:min_len]
            
            self.weights = self._optimize_weights_optuna(
                X_val[:min_len], adjusted_y_val, adjusted_predictions
            )
        
        print("Treinamento do ensemble concluído!")
        return self
    
    def predict(self, data: pd.DataFrame, days_ahead: int = 30) -> Dict[str, Any]:
        """
        Faz previsões usando o ensemble de modelos.
        
        Args:
            data (pd.DataFrame): DataFrame com dados históricos
            days_ahead (int): Número de dias para prever
            
        Returns:
            Dict[str, Any]: Previsões do ensemble e modelos individuais
        """
        # Prepara features
        featured_data = self._prepare_features(data)
        
        # Coleta previsões de cada modelo
        individual_predictions = {}
        
        # LSTM
        try:
            lstm_pred = self.models['lstm'].predict(data, days_ahead)
            individual_predictions['lstm'] = lstm_pred
        except Exception as e:
            print(f"Erro no LSTM: {e}")
        
        # Random Forest
        try:
            rf_pred = self.models['random_forest'].predict(featured_data, days_ahead)
            individual_predictions['random_forest'] = rf_pred
        except Exception as e:
            print(f"Erro no Random Forest: {e}")
        
        # LightGBM
        try:
            lgb_pred = self.models['lightgbm'].predict(featured_data, days_ahead)
            individual_predictions['lightgbm'] = lgb_pred
        except Exception as e:
            print(f"Erro no LightGBM: {e}")
        
        # XGBoost
        try:
            if self.models['xgboost'] is not None:
                feature_cols = [col for col in featured_data.columns 
                              if col not in ['Close', 'Open', 'High', 'Low', 'Volume']]
                
                # Para XGBoost, fazemos previsão iterativa
                last_features = featured_data[feature_cols].iloc[-1:].values
                xgb_predictions = []
                
                for _ in range(days_ahead):
                    pred = self.models['xgboost'].predict(last_features)[0]
                    xgb_predictions.append(pred)
                    # Atualiza features para próxima previsão (simplificado)
                    # Em implementação real, seria mais sofisticado
                
                individual_predictions['xgboost'] = {
                    'predictions': xgb_predictions,
                    'ticker': data.name if hasattr(data, 'name') else 'unknown'
                }
        except Exception as e:
            print(f"Erro no XGBoost: {e}")
        
        # Combina previsões usando ensemble
        ensemble_predictions = self._combine_predictions(individual_predictions, days_ahead)
        
        return {
            'ensemble': ensemble_predictions,
            'individual': individual_predictions,
            'weights': self.weights,
            'metrics': self.individual_metrics
        }
    
    def _combine_predictions(self, individual_predictions: Dict[str, Any], 
                           days_ahead: int) -> Dict[str, Any]:
        """
        Combina previsões individuais usando o método de ensemble escolhido.
        
        Args:
            individual_predictions (Dict[str, Any]): Previsões individuais
            days_ahead (int): Número de dias previstos
            
        Returns:
            Dict[str, Any]: Previsões combinadas
        """
        if self.ensemble_method == 'weighted_average':
            return self._weighted_average_ensemble(individual_predictions, days_ahead)
        elif self.ensemble_method == 'voting':
            return self._voting_ensemble(individual_predictions, days_ahead)
        elif self.ensemble_method == 'stacking':
            return self._stacking_ensemble(individual_predictions, days_ahead)
        else:
            raise ValueError(f"Método de ensemble não suportado: {self.ensemble_method}")
    
    def _weighted_average_ensemble(self, individual_predictions: Dict[str, Any], 
                                 days_ahead: int) -> Dict[str, Any]:
        """
        Combina previsões usando média ponderada.
        """
        ensemble_preds = []
        dates = []
        
        # Extrai previsões de cada modelo
        model_preds = {}
        for model_name, pred_data in individual_predictions.items():
            if 'predictions' in pred_data:
                if isinstance(pred_data['predictions'], list) and len(pred_data['predictions']) > 0:
                    if isinstance(pred_data['predictions'][0], dict):
                        # Formato LSTM
                        model_preds[model_name] = [p['predicted_price'] for p in pred_data['predictions']]
                        if not dates:
                            dates = [p['date'] for p in pred_data['predictions']]
                    else:
                        # Formato tree-based
                        model_preds[model_name] = pred_data['predictions']
        
        # Combina usando pesos
        for i in range(days_ahead):
            weighted_pred = 0
            total_weight = 0
            
            for model_name, preds in model_preds.items():
                if i < len(preds) and model_name in self.weights:
                    weight = self.weights[model_name]
                    weighted_pred += weight * preds[i]
                    total_weight += weight
            
            if total_weight > 0:
                ensemble_preds.append(weighted_pred / total_weight)
            else:
                ensemble_preds.append(0)
        
        # Gera datas se não existirem
        if not dates:
            from datetime import datetime, timedelta
            base_date = datetime.now()
            dates = [(base_date + timedelta(days=i)).strftime('%Y-%m-%d') 
                    for i in range(1, days_ahead + 1)]
        
        return {
            'predictions': [
                {
                    'date': dates[i] if i < len(dates) else f'day_{i+1}',
                    'predicted_price': float(ensemble_preds[i]),
                    'confidence': self._calculate_confidence(individual_predictions, i)
                }
                for i in range(len(ensemble_preds))
            ],
            'method': 'weighted_average',
            'weights_used': self.weights
        }
    
    def _voting_ensemble(self, individual_predictions: Dict[str, Any], 
                        days_ahead: int) -> Dict[str, Any]:
        """
        Combina previsões usando votação (média simples).
        """
        # Implementação similar à média ponderada, mas com pesos iguais
        equal_weights = {model: 1.0/len(individual_predictions) 
                        for model in individual_predictions.keys()}
        
        original_weights = self.weights.copy()
        self.weights = equal_weights
        
        result = self._weighted_average_ensemble(individual_predictions, days_ahead)
        result['method'] = 'voting'
        
        self.weights = original_weights
        return result
    
    def _stacking_ensemble(self, individual_predictions: Dict[str, Any], 
                          days_ahead: int) -> Dict[str, Any]:
        """
        Combina previsões usando stacking (meta-modelo).
        """
        # Para implementação completa de stacking, seria necessário treinar um meta-modelo
        # Por simplicidade, usa média ponderada como fallback
        return self._weighted_average_ensemble(individual_predictions, days_ahead)
    
    def _calculate_confidence(self, individual_predictions: Dict[str, Any], 
                            day_index: int) -> float:
        """
        Calcula confiança baseada na concordância entre modelos.
        """
        predictions_for_day = []
        
        for model_name, pred_data in individual_predictions.items():
            if 'predictions' in pred_data:
                preds = pred_data['predictions']
                if isinstance(preds, list) and day_index < len(preds):
                    if isinstance(preds[day_index], dict):
                        predictions_for_day.append(preds[day_index]['predicted_price'])
                    else:
                        predictions_for_day.append(preds[day_index])
        
        if len(predictions_for_day) < 2:
            return 0.5
        
        # Calcula coeficiente de variação (inverso da confiança)
        mean_pred = np.mean(predictions_for_day)
        std_pred = np.std(predictions_for_day)
        
        if mean_pred == 0:
            return 0.5
        
        cv = std_pred / abs(mean_pred)
        confidence = max(0.1, min(0.9, 1 - cv))
        
        return float(confidence)
    
    def evaluate(self, test_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Avalia o ensemble em dados de teste.
        
        Args:
            test_data (pd.DataFrame): Dados de teste
            
        Returns:
            Dict[str, Any]: Métricas de avaliação
        """
        # Faz previsões
        predictions = self.predict(test_data, days_ahead=len(test_data))
        
        # Extrai previsões ensemble
        ensemble_preds = [p['predicted_price'] for p in predictions['ensemble']['predictions']]
        y_true = test_data['Close'].values
        
        # Ajusta tamanhos
        min_len = min(len(ensemble_preds), len(y_true))
        ensemble_preds = ensemble_preds[:min_len]
        y_true = y_true[:min_len]
        
        # Calcula métricas
        mse = mean_squared_error(y_true, ensemble_preds)
        mae = mean_absolute_error(y_true, ensemble_preds)
        rmse = np.sqrt(mse)
        
        # Calcula acurácia direcional
        if len(y_true) > 1:
            direction_true = np.diff(y_true) > 0
            direction_pred = np.diff(ensemble_preds) > 0
            direction_accuracy = np.mean(direction_true == direction_pred) * 100
        else:
            direction_accuracy = 0
        
        return {
            'ensemble_metrics': {
                'mse': float(mse),
                'mae': float(mae),
                'rmse': float(rmse),
                'direction_accuracy': float(direction_accuracy)
            },
            'individual_metrics': self.individual_metrics,
            'weights': self.weights
        }
    
    def save(self, model_path: str) -> str:
        """
        Salva o ensemble completo.
        
        Args:
            model_path (str): Caminho base para salvar
            
        Returns:
            str: Caminho onde foi salvo
        """
        os.makedirs(model_path, exist_ok=True)
        
        # Salva cada modelo individual
        for model_name, model in self.models.items():
            if model is not None and hasattr(model, 'save'):
                model.save(os.path.join(model_path, f"{model_name}_model"))
            elif model_name == 'xgboost' and model is not None:
                joblib.dump(model, os.path.join(model_path, f"{model_name}_model.pkl"))
        
        # Salva configurações do ensemble
        ensemble_config = {
            'ensemble_method': self.ensemble_method,
            'weights': self.weights,
            'individual_metrics': self.individual_metrics,
            'optimization_history': self.optimization_history
        }
        
        joblib.dump(ensemble_config, os.path.join(model_path, 'ensemble_config.pkl'))
        
        return model_path
    
    def load(self, model_path: str) -> 'EnsembleModel':
        """
        Carrega o ensemble completo.
        
        Args:
            model_path (str): Caminho base para carregar
            
        Returns:
            EnsembleModel: Instância carregada
        """
        # Carrega configurações do ensemble
        config_path = os.path.join(model_path, 'ensemble_config.pkl')
        if os.path.exists(config_path):
            config = joblib.load(config_path)
            self.ensemble_method = config.get('ensemble_method', self.ensemble_method)
            self.weights = config.get('weights', self.weights)
            self.individual_metrics = config.get('individual_metrics', {})
            self.optimization_history = config.get('optimization_history', [])
        
        # Carrega modelos individuais
        for model_name in self.models.keys():
            model_file = os.path.join(model_path, f"{model_name}_model")
            
            if model_name == 'xgboost':
                pkl_file = f"{model_file}.pkl"
                if os.path.exists(pkl_file):
                    self.models[model_name] = joblib.load(pkl_file)
            else:
                if hasattr(self.models[model_name], 'load'):
                    try:
                        self.models[model_name].load(model_file)
                    except:
                        print(f"Não foi possível carregar o modelo {model_name}")
        
        return self

