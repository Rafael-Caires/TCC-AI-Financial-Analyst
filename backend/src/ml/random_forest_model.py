"""
Módulo de machine learning para modelo Random Forest

Este arquivo implementa um modelo Random Forest para previsão
de séries temporais financeiras.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime, timedelta

class RandomForestModel:
    """
    Classe para criação, treinamento e previsão usando modelo Random Forest
    para séries temporais financeiras.
    """
    
    def __init__(self, n_estimators=100, max_depth=None, sequence_length=10, model_path=None):
        """
        Inicializa o modelo Random Forest.
        
        Args:
            n_estimators (int): Número de árvores na floresta
            max_depth (int): Profundidade máxima das árvores
            sequence_length (int): Tamanho da sequência para entrada do modelo
            model_path (str): Caminho para salvar/carregar o modelo
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.sequence_length = sequence_length
        self.model_path = model_path
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        
    def _create_features(self, data):
        """
        Cria features para o modelo Random Forest.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados de preços
            
        Returns:
            tuple: (X, y) onde X são as features e y são os valores alvo
        """
        df = data.copy()
        
        # Adiciona features técnicas
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        
        # Retornos percentuais
        df['Return_1d'] = df['Close'].pct_change(1)
        df['Return_5d'] = df['Close'].pct_change(5)
        df['Return_10d'] = df['Close'].pct_change(10)
        
        # Volatilidade
        df['Volatility_5d'] = df['Return_1d'].rolling(window=5).std()
        df['Volatility_20d'] = df['Return_1d'].rolling(window=20).std()
        
        # Volume
        df['Volume_Change'] = df['Volume'].pct_change(1)
        df['Volume_SMA_5'] = df['Volume'].rolling(window=5).mean()
        
        # Indicadores de tendência
        df['Price_SMA5_Ratio'] = df['Close'] / df['SMA_5']
        df['Price_SMA20_Ratio'] = df['Close'] / df['SMA_20']
        df['SMA5_SMA20_Ratio'] = df['SMA_5'] / df['SMA_20']
        
        # Remove linhas com NaN
        df.dropna(inplace=True)
        
        # Cria sequências para cada dia
        X, y = [], []
        for i in range(len(df) - self.sequence_length):
            # Seleciona as features para a sequência atual
            features = df.iloc[i:i+self.sequence_length]
            
            # Extrai as colunas relevantes
            feature_cols = [
                'Close', 'SMA_5', 'SMA_20', 'SMA_50', 
                'Return_1d', 'Return_5d', 'Return_10d',
                'Volatility_5d', 'Volatility_20d',
                'Volume_Change', 'Volume_SMA_5',
                'Price_SMA5_Ratio', 'Price_SMA20_Ratio', 'SMA5_SMA20_Ratio'
            ]
            
            # Flatten para criar um vetor de features
            X.append(features[feature_cols].values.flatten())
            
            # O alvo é o preço de fechamento do próximo dia
            y.append(df.iloc[i+self.sequence_length]['Close'])
        
        return np.array(X), np.array(y)
    
    def train(self, data, test_size=0.2, verbose=1):
        """
        Treina o modelo Random Forest com os dados fornecidos.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados de preços
            test_size (float): Proporção dos dados para teste
            verbose (int): Nível de verbosidade durante o treinamento
            
        Returns:
            self: Instância do modelo treinado
        """
        # Cria features
        X, y = self._create_features(data)
        
        # Normaliza os dados
        X_scaled = self.scaler.fit_transform(X)
        
        # Divide os dados em treinamento e teste
        train_size = int(len(X_scaled) * (1 - test_size))
        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Treina o modelo
        if verbose:
            print(f"Treinando modelo Random Forest com {self.n_estimators} árvores...")
        
        self.model.fit(X_train, y_train)
        
        # Avalia o modelo no conjunto de teste
        if verbose:
            y_pred = self.model.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            
            print(f"Avaliação no conjunto de teste:")
            print(f"MSE: {mse:.4f}")
            print(f"RMSE: {rmse:.4f}")
            print(f"MAE: {mae:.4f}")
        
        return self
    
    def predict(self, data, days_ahead=30):
        """
        Faz previsões para os próximos dias.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados históricos
            days_ahead (int): Número de dias para prever à frente
            
        Returns:
            dict: Dicionário com previsões, datas e intervalos de confiança
        """
        # Cria features para os dados históricos
        X, _ = self._create_features(data)
        
        if len(X) == 0:
            raise ValueError("Não há dados suficientes para criar features.")
        
        # Normaliza os dados
        X_scaled = self.scaler.transform(X)
        
        # Inicializa listas para armazenar resultados
        predictions = []
        prediction_dates = []
        lower_bounds = []
        upper_bounds = []
        
        # Data inicial para previsões (dia seguinte ao último dado)
        current_date = data.index[-1]
        
        # Dados atuais para previsão
        current_data = data.copy()
        
        # Faz previsões para cada dia futuro
        for i in range(days_ahead):
            # Avança a data
            current_date = self._next_business_day(current_date)
            prediction_dates.append(current_date)
            
            # Cria features para a previsão atual
            X_current, _ = self._create_features(current_data)
            X_current_scaled = self.scaler.transform(X_current[-1:])
            
            # Faz a previsão
            predicted_price = self.model.predict(X_current_scaled)[0]
            
            # Calcula intervalo de confiança usando a variância das previsões das árvores
            tree_predictions = [tree.predict(X_current_scaled)[0] for tree in self.model.estimators_]
            std_dev = np.std(tree_predictions)
            
            # Intervalo de confiança de 95% (aproximadamente 2 desvios padrão)
            lower_bound = max(0, predicted_price - 2 * std_dev)
            upper_bound = predicted_price + 2 * std_dev
            
            # Armazena os resultados
            predictions.append(predicted_price)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            
            # Adiciona a previsão aos dados para a próxima iteração
            new_row = pd.DataFrame({
                'Open': [predicted_price],
                'High': [predicted_price * 1.01],  # Simulação simples
                'Low': [predicted_price * 0.99],   # Simulação simples
                'Close': [predicted_price],
                'Volume': [current_data['Volume'].mean()]  # Usa a média do volume
            }, index=[current_date])
            
            current_data = pd.concat([current_data, new_row])
        
        # Formata os resultados
        results = []
        for i in range(days_ahead):
            results.append({
                'date': prediction_dates[i].strftime('%Y-%m-%d'),
                'predicted_price': float(predictions[i]),
                'lower_bound': float(lower_bounds[i]),
                'upper_bound': float(upper_bounds[i]),
            })
        
        return {
            'ticker': data.name if hasattr(data, 'name') else 'unknown',
            'last_price': float(data['Close'].iloc[-1]),
            'forecast_days': days_ahead,
            'predictions': results
        }
    
    def evaluate(self, test_data):
        """
        Avalia o modelo em dados de teste.
        
        Args:
            test_data (pandas.DataFrame): DataFrame com dados de teste
            
        Returns:
            dict: Métricas de avaliação
        """
        # Cria features para os dados de teste
        X_test, y_test = self._create_features(test_data)
        
        # Normaliza os dados
        X_test_scaled = self.scaler.transform(X_test)
        
        # Faz previsões
        y_pred = self.model.predict(X_test_scaled)
        
        # Calcula métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        # Calcula acurácia direcional
        direction_true = np.diff(y_test)
        direction_pred = np.diff(y_pred)
        direction_accuracy = np.mean((direction_true > 0) == (direction_pred > 0)) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy)
        }
    
    def save(self, model_path=None):
        """
        Salva o modelo treinado e o scaler.
        
        Args:
            model_path (str): Caminho para salvar o modelo (opcional)
            
        Returns:
            str: Caminho onde o modelo foi salvo
        """
        # Usa o caminho fornecido ou o padrão
        save_path = model_path or self.model_path
        
        if save_path is None:
            # Cria um nome baseado na data e hora atual
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"rf_model_{timestamp}"
        
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Salva o modelo
        joblib.dump(self.model, f"{save_path}.pkl")
        
        # Salva o scaler
        joblib.dump(self.scaler, f"{save_path}_scaler.pkl")
        
        return save_path
    
    def load(self, model_path=None):
        """
        Carrega um modelo treinado e o scaler.
        
        Args:
            model_path (str): Caminho para carregar o modelo (opcional)
            
        Returns:
            self: Instância com o modelo carregado
        """
        # Usa o caminho fornecido ou o padrão
        load_path = model_path or self.model_path
        
        if load_path is None:
            raise ValueError("É necessário fornecer um caminho para carregar o modelo.")
        
        # Carrega o modelo
        self.model = joblib.load(f"{load_path}.pkl")
        
        # Carrega o scaler
        self.scaler = joblib.load(f"{load_path}_scaler.pkl")
        
        return self
    
    def plot_feature_importance(self, data):
        """
        Plota a importância das features do modelo.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados para obter nomes das features
            
        Returns:
            matplotlib.figure.Figure: Figura com o gráfico de importância
        """
        # Cria features para obter os nomes
        X, _ = self._create_features(data)
        
        # Obtém os nomes das features
        feature_cols = []
        for i in range(self.sequence_length):
            day = self.sequence_length - i
            feature_cols.extend([
                f'Close (t-{day})', f'SMA_5 (t-{day})', f'SMA_20 (t-{day})', f'SMA_50 (t-{day})', 
                f'Return_1d (t-{day})', f'Return_5d (t-{day})', f'Return_10d (t-{day})',
                f'Volatility_5d (t-{day})', f'Volatility_20d (t-{day})',
                f'Volume_Change (t-{day})', f'Volume_SMA_5 (t-{day})',
                f'Price_SMA5_Ratio (t-{day})', f'Price_SMA20_Ratio (t-{day})', f'SMA5_SMA20_Ratio (t-{day})'
            ])
        
        # Obtém a importância das features
        importances = self.model.feature_importances_
        
        # Cria um DataFrame com as importâncias
        feature_importance = pd.DataFrame({
            'Feature': feature_cols[:len(importances)],
            'Importance': importances
        })
        
        # Ordena por importância
        feature_importance = feature_importance.sort_values('Importance', ascending=False)
        
        # Plota as 20 features mais importantes
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.barh(feature_importance['Feature'][:20], feature_importance['Importance'][:20])
        ax.set_title('Importância das Features - Random Forest')
        ax.set_xlabel('Importância')
        plt.tight_layout()
        
        return fig
    
    def plot_predictions(self, data, predictions):
        """
        Plota os dados históricos e as previsões.
        
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
        ax.plot(dates, predicted_prices, label='Previsões', color='green', linestyle='--')
        
        # Plota o intervalo de confiança
        ax.fill_between(dates, lower_bounds, upper_bounds, color='green', alpha=0.2, label='Intervalo de Confiança')
        
        # Configurações do gráfico
        ax.set_title(f"Previsão de Preços para {predictions['ticker']} (Random Forest)")
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço')
        ax.legend()
        ax.grid(True)
        
        # Formata o eixo x para mostrar datas
        fig.autofmt_xdate()
        
        return fig
    
    def _next_business_day(self, date):
        """
        Retorna o próximo dia útil (ignorando fins de semana).
        
        Args:
            date (datetime): Data atual
            
        Returns:
            datetime: Próximo dia útil
        """
        next_day = date + timedelta(days=1)
        
        # Pula fins de semana
        while next_day.weekday() >= 5:  # 5 = Sábado, 6 = Domingo
            next_day += timedelta(days=1)
            
        return next_day
