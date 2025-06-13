"""
Módulo de machine learning para modelo LSTM

Este arquivo implementa um modelo LSTM (Long Short-Term Memory) para previsão
de séries temporais financeiras.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
import joblib
from datetime import datetime, timedelta

class LSTMModel:
    """
    Classe para criação, treinamento e previsão usando modelo LSTM
    para séries temporais financeiras.
    """
    
    def __init__(self, sequence_length=60, epochs=50, batch_size=32, model_path=None):
        """
        Inicializa o modelo LSTM.
        
        Args:
            sequence_length (int): Tamanho da sequência para entrada do LSTM
            epochs (int): Número máximo de épocas para treinamento
            batch_size (int): Tamanho do lote para treinamento
            model_path (str): Caminho para salvar/carregar o modelo
        """
        self.sequence_length = sequence_length
        self.epochs = epochs
        self.batch_size = batch_size
        self.model_path = model_path
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.history = None
        
    def _create_sequences(self, data):
        """
        Cria sequências para treinamento do modelo LSTM.
        
        Args:
            data (numpy.ndarray): Dados normalizados
            
        Returns:
            tuple: (X, y) onde X são as sequências de entrada e y são os valores alvo
        """
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:(i + self.sequence_length)])
            y.append(data[i + self.sequence_length])
        return np.array(X), np.array(y)
    
    def _build_model(self, input_shape):
        """
        Constrói a arquitetura do modelo LSTM.
        
        Args:
            input_shape (tuple): Forma dos dados de entrada (sequence_length, features)
            
        Returns:
            tensorflow.keras.models.Sequential: Modelo LSTM construído
        """
        model = Sequential()
        
        # Primeira camada LSTM com retorno de sequências
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Segunda camada LSTM
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Camada densa para reduzir dimensionalidade
        model.add(Dense(units=25))
        
        # Camada de saída
        model.add(Dense(units=1))
        
        # Compilação do modelo
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        return model
    
    def train(self, data, validation_split=0.2, verbose=1):
        """
        Treina o modelo LSTM com os dados fornecidos.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados de preços
            validation_split (float): Proporção dos dados para validação
            verbose (int): Nível de verbosidade durante o treinamento
            
        Returns:
            self: Instância do modelo treinado
        """
        # Extrai a coluna de preços de fechamento
        prices = data['Close'].values.reshape(-1, 1)
        
        # Normaliza os dados
        scaled_data = self.scaler.fit_transform(prices)
        
        # Cria sequências para treinamento
        X, y = self._create_sequences(scaled_data)
        
        # Divide os dados em treinamento e validação
        train_size = int(len(X) * (1 - validation_split))
        X_train, X_val = X[:train_size], X[train_size:]
        y_train, y_val = y[:train_size], y[train_size:]
        
        # Constrói o modelo
        self.model = self._build_model((X_train.shape[1], X_train.shape[2]))
        
        # Configura early stopping para evitar overfitting
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True
        )
        
        # Treina o modelo
        self.history = self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_data=(X_val, y_val),
            callbacks=[early_stopping],
            verbose=verbose
        )
        
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
        if self.model is None:
            raise ValueError("O modelo precisa ser treinado ou carregado antes de fazer previsões.")
        
        # Extrai a coluna de preços de fechamento
        prices = data['Close'].values.reshape(-1, 1)
        
        # Normaliza os dados
        scaled_data = self.scaler.transform(prices)
        
        # Prepara a sequência inicial para previsão
        last_sequence = scaled_data[-self.sequence_length:]
        
        # Inicializa listas para armazenar resultados
        predictions = []
        prediction_dates = []
        lower_bounds = []
        upper_bounds = []
        
        # Data inicial para previsões (dia seguinte ao último dado)
        current_date = data.index[-1]
        
        # Sequência atual para previsão
        current_sequence = last_sequence.reshape(1, self.sequence_length, 1)
        
        # Faz previsões para cada dia futuro
        for i in range(days_ahead):
            # Avança a data
            current_date = self._next_business_day(current_date)
            prediction_dates.append(current_date)
            
            # Faz a previsão
            predicted_scaled = self.model.predict(current_sequence, verbose=0)[0][0]
            
            # Adiciona alguma incerteza (simulação de intervalo de confiança)
            volatility = np.std(scaled_data[-30:]) * 2  # Usa a volatilidade recente
            lower_scaled = max(0, predicted_scaled - volatility)
            upper_scaled = predicted_scaled + volatility
            
            # Desnormaliza as previsões
            predicted_price = self.scaler.inverse_transform(np.array([[predicted_scaled]]))[0][0]
            lower_bound = self.scaler.inverse_transform(np.array([[lower_scaled]]))[0][0]
            upper_bound = self.scaler.inverse_transform(np.array([[upper_scaled]]))[0][0]
            
            # Armazena os resultados
            predictions.append(predicted_price)
            lower_bounds.append(lower_bound)
            upper_bounds.append(upper_bound)
            
            # Atualiza a sequência para a próxima previsão
            new_sequence = np.append(current_sequence[0, 1:, :], [[predicted_scaled]], axis=0)
            current_sequence = new_sequence.reshape(1, self.sequence_length, 1)
        
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
            'last_price': float(prices[-1][0]),
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
        if self.model is None:
            raise ValueError("O modelo precisa ser treinado ou carregado antes de ser avaliado.")
        
        # Extrai a coluna de preços de fechamento
        prices = test_data['Close'].values.reshape(-1, 1)
        
        # Normaliza os dados
        scaled_data = self.scaler.transform(prices)
        
        # Cria sequências para teste
        X_test, y_test = self._create_sequences(scaled_data)
        
        # Avalia o modelo
        loss = self.model.evaluate(X_test, y_test, verbose=0)
        
        # Faz previsões
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        
        # Desnormaliza as previsões
        y_pred = self.scaler.inverse_transform(y_pred_scaled)
        y_true = self.scaler.inverse_transform(y_test.reshape(-1, 1))
        
        # Calcula métricas
        mse = np.mean(np.square(y_pred - y_true))
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(y_pred - y_true))
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        # Calcula acurácia direcional
        direction_true = np.diff(y_true.flatten())
        direction_pred = np.diff(y_pred.flatten())
        direction_accuracy = np.mean((direction_true > 0) == (direction_pred > 0)) * 100
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'mape': float(mape),
            'direction_accuracy': float(direction_accuracy),
            'loss': float(loss)
        }
    
    def save(self, model_path=None):
        """
        Salva o modelo treinado e o scaler.
        
        Args:
            model_path (str): Caminho para salvar o modelo (opcional)
            
        Returns:
            str: Caminho onde o modelo foi salvo
        """
        if self.model is None:
            raise ValueError("O modelo precisa ser treinado antes de ser salvo.")
        
        # Usa o caminho fornecido ou o padrão
        save_path = model_path or self.model_path
        
        if save_path is None:
            # Cria um nome baseado na data e hora atual
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"lstm_model_{timestamp}"
        
        # Cria o diretório se não existir
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Salva o modelo
        self.model.save(f"{save_path}.h5")
        
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
        self.model = tf.keras.models.load_model(f"{load_path}.h5")
        
        # Carrega o scaler
        self.scaler = joblib.load(f"{load_path}_scaler.pkl")
        
        return self
    
    def plot_training_history(self):
        """
        Plota o histórico de treinamento do modelo.
        
        Returns:
            matplotlib.figure.Figure: Figura com o gráfico do histórico
        """
        if self.history is None:
            raise ValueError("O modelo precisa ser treinado antes de plotar o histórico.")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plota a perda de treinamento e validação
        ax.plot(self.history.history['loss'], label='Treinamento')
        ax.plot(self.history.history['val_loss'], label='Validação')
        
        ax.set_title('Histórico de Treinamento do Modelo LSTM')
        ax.set_xlabel('Época')
        ax.set_ylabel('Perda (MSE)')
        ax.legend()
        ax.grid(True)
        
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
        ax.plot(dates, predicted_prices, label='Previsões', color='red', linestyle='--')
        
        # Plota o intervalo de confiança
        ax.fill_between(dates, lower_bounds, upper_bounds, color='red', alpha=0.2, label='Intervalo de Confiança')
        
        # Configurações do gráfico
        ax.set_title(f"Previsão de Preços para {predictions['ticker']}")
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
