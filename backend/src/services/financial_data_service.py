"""
Serviço para manipulação de dados financeiros

Este arquivo implementa serviços para obtenção, processamento e análise
de dados financeiros.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import io
import base64

class FinancialDataService:
    """
    Serviço para manipulação de dados financeiros.
    """
    
    @staticmethod
    def get_stock_data(ticker, start_date=None, end_date=None, period=None):
        """
        Obtém dados históricos de um ativo.
        
        Args:
            ticker (str): Símbolo do ativo
            start_date (str): Data de início (formato: 'YYYY-MM-DD')
            end_date (str): Data de fim (formato: 'YYYY-MM-DD')
            period (str): Período alternativo ('1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', '10y', 'ytd', 'max')
            
        Returns:
            pandas.DataFrame: DataFrame com dados históricos
        """
        # Configura datas padrão se não fornecidas
        if not end_date:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        if not start_date and not period:
            # Padrão: 1 ano atrás
            start_date = (datetime.now() - timedelta(days=365)).strftime('%Y-%m-%d')
        
        try:
            # Obtém dados do Yahoo Finance
            if period:
                data = yf.download(ticker, period=period)
            else:
                data = yf.download(ticker, start=start_date, end=end_date)
            
            # Adiciona o ticker como nome do DataFrame
            data.name = ticker
            
            return data
        except Exception as e:
            raise Exception(f"Erro ao obter dados para {ticker}: {str(e)}")
    
    @staticmethod
    def calculate_technical_indicators(data):
        """
        Calcula indicadores técnicos para os dados fornecidos.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados históricos
            
        Returns:
            pandas.DataFrame: DataFrame com indicadores técnicos adicionados
        """
        df = data.copy()
        
        # Médias Móveis Simples (SMA)
        df['SMA_5'] = df['Close'].rolling(window=5).mean()
        df['SMA_20'] = df['Close'].rolling(window=20).mean()
        df['SMA_50'] = df['Close'].rolling(window=50).mean()
        df['SMA_200'] = df['Close'].rolling(window=200).mean()
        
        # Médias Móveis Exponenciais (EMA)
        df['EMA_5'] = df['Close'].ewm(span=5, adjust=False).mean()
        df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
        df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
        
        # MACD (Moving Average Convergence Divergence)
        df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
        df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
        df['MACD'] = df['EMA_12'] - df['EMA_26']
        df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
        df['MACD_Histogram'] = df['MACD'] - df['MACD_Signal']
        
        # RSI (Relative Strength Index)
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=14).mean()
        avg_loss = loss.rolling(window=14).mean()
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=20).mean()
        df['BB_Std'] = df['Close'].rolling(window=20).std()
        df['BB_Upper'] = df['BB_Middle'] + (df['BB_Std'] * 2)
        df['BB_Lower'] = df['BB_Middle'] - (df['BB_Std'] * 2)
        
        # Stochastic Oscillator
        low_14 = df['Low'].rolling(window=14).min()
        high_14 = df['High'].rolling(window=14).max()
        df['Stoch_K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
        df['Stoch_D'] = df['Stoch_K'].rolling(window=3).mean()
        
        # Average True Range (ATR)
        tr1 = df['High'] - df['Low']
        tr2 = abs(df['High'] - df['Close'].shift())
        tr3 = abs(df['Low'] - df['Close'].shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=14).mean()
        
        # On-Balance Volume (OBV)
        df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
        
        return df
    
    @staticmethod
    def normalize_data(data, columns=None):
        """
        Normaliza os dados para valores entre 0 e 1.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados históricos
            columns (list): Lista de colunas para normalizar (None = todas)
            
        Returns:
            pandas.DataFrame: DataFrame com dados normalizados
        """
        df = data.copy()
        
        if columns is None:
            # Normaliza apenas colunas numéricas
            columns = df.select_dtypes(include=[np.number]).columns
        
        for column in columns:
            if column in df.columns:
                min_val = df[column].min()
                max_val = df[column].max()
                
                # Evita divisão por zero
                if max_val > min_val:
                    df[column] = (df[column] - min_val) / (max_val - min_val)
        
        return df
    
    @staticmethod
    def generate_price_chart(data, ticker, indicators=None):
        """
        Gera um gráfico de preços com indicadores técnicos.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados históricos
            ticker (str): Símbolo do ativo
            indicators (list): Lista de indicadores para incluir
            
        Returns:
            str: Imagem do gráfico em formato base64
        """
        # Configura o gráfico
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plota o preço de fechamento
        ax.plot(data.index, data['Close'], label='Preço de Fechamento', color='blue')
        
        # Adiciona indicadores selecionados
        if indicators:
            if 'SMA_20' in indicators and 'SMA_20' in data.columns:
                ax.plot(data.index, data['SMA_20'], label='SMA 20', color='red', linestyle='--')
            
            if 'SMA_50' in indicators and 'SMA_50' in data.columns:
                ax.plot(data.index, data['SMA_50'], label='SMA 50', color='green', linestyle='--')
            
            if 'SMA_200' in indicators and 'SMA_200' in data.columns:
                ax.plot(data.index, data['SMA_200'], label='SMA 200', color='purple', linestyle='--')
            
            if 'BB' in indicators and 'BB_Upper' in data.columns and 'BB_Lower' in data.columns:
                ax.plot(data.index, data['BB_Upper'], label='Bollinger Superior', color='gray', linestyle=':')
                ax.plot(data.index, data['BB_Lower'], label='Bollinger Inferior', color='gray', linestyle=':')
                ax.fill_between(data.index, data['BB_Upper'], data['BB_Lower'], color='gray', alpha=0.1)
        
        # Configurações do gráfico
        ax.set_title(f'Histórico de Preços - {ticker}')
        ax.set_xlabel('Data')
        ax.set_ylabel('Preço')
        ax.legend()
        ax.grid(True)
        
        # Formata o eixo x para mostrar datas
        fig.autofmt_xdate()
        
        # Converte o gráfico para base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return image_base64
    
    @staticmethod
    def generate_technical_chart(data, ticker, indicator):
        """
        Gera um gráfico de indicador técnico.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados históricos
            ticker (str): Símbolo do ativo
            indicator (str): Indicador técnico para plotar
            
        Returns:
            str: Imagem do gráfico em formato base64
        """
        # Configura o gráfico
        fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
        
        # Plota o preço de fechamento no gráfico superior
        axes[0].plot(data.index, data['Close'], label='Preço de Fechamento', color='blue')
        axes[0].set_title(f'Histórico de Preços - {ticker}')
        axes[0].set_ylabel('Preço')
        axes[0].legend()
        axes[0].grid(True)
        
        # Plota o indicador técnico no gráfico inferior
        if indicator == 'RSI':
            axes[1].plot(data.index, data['RSI'], label='RSI', color='purple')
            axes[1].axhline(y=70, color='red', linestyle='--')
            axes[1].axhline(y=30, color='green', linestyle='--')
            axes[1].set_ylim(0, 100)
            axes[1].set_ylabel('RSI')
        
        elif indicator == 'MACD':
            axes[1].plot(data.index, data['MACD'], label='MACD', color='blue')
            axes[1].plot(data.index, data['MACD_Signal'], label='Sinal', color='red')
            axes[1].bar(data.index, data['MACD_Histogram'], label='Histograma', color='gray', alpha=0.3)
            axes[1].set_ylabel('MACD')
        
        elif indicator == 'Stochastic':
            axes[1].plot(data.index, data['Stoch_K'], label='%K', color='blue')
            axes[1].plot(data.index, data['Stoch_D'], label='%D', color='red')
            axes[1].axhline(y=80, color='red', linestyle='--')
            axes[1].axhline(y=20, color='green', linestyle='--')
            axes[1].set_ylim(0, 100)
            axes[1].set_ylabel('Estocástico')
        
        elif indicator == 'Volume':
            axes[1].bar(data.index, data['Volume'], label='Volume', color='blue', alpha=0.3)
            axes[1].set_ylabel('Volume')
        
        axes[1].set_xlabel('Data')
        axes[1].legend()
        axes[1].grid(True)
        
        # Formata o eixo x para mostrar datas
        fig.autofmt_xdate()
        
        # Ajusta o layout
        plt.tight_layout()
        
        # Converte o gráfico para base64
        buffer = io.BytesIO()
        fig.savefig(buffer, format='png')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        
        return image_base64
    
    @staticmethod
    def calculate_returns(data, periods=[1, 5, 20, 60, 252]):
        """
        Calcula retornos para diferentes períodos.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados históricos
            periods (list): Lista de períodos para calcular retornos
            
        Returns:
            dict: Dicionário com retornos calculados
        """
        returns = {}
        
        for period in periods:
            if len(data) > period:
                # Retorno percentual
                returns[f'{period}d'] = ((data['Close'].iloc[-1] / data['Close'].iloc[-period-1]) - 1) * 100
            else:
                returns[f'{period}d'] = None
        
        return returns
    
    @staticmethod
    def calculate_volatility(data, periods=[20, 60, 252]):
        """
        Calcula volatilidade para diferentes períodos.
        
        Args:
            data (pandas.DataFrame): DataFrame com dados históricos
            periods (list): Lista de períodos para calcular volatilidade
            
        Returns:
            dict: Dicionário com volatilidades calculadas
        """
        volatility = {}
        
        for period in periods:
            if len(data) > period:
                # Volatilidade (desvio padrão dos retornos diários * raiz do número de dias de negociação)
                daily_returns = data['Close'].pct_change().dropna()
                volatility[f'{period}d'] = daily_returns.tail(period).std() * np.sqrt(252) * 100
            else:
                volatility[f'{period}d'] = None
        
        return volatility
    
    @staticmethod
    def get_stock_info(ticker):
        """
        Obtém informações gerais sobre um ativo.
        
        Args:
            ticker (str): Símbolo do ativo
            
        Returns:
            dict: Dicionário com informações do ativo
        """
        try:
            # Obtém informações do Yahoo Finance
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Seleciona informações relevantes
            relevant_info = {
                'ticker': ticker,
                'name': info.get('shortName', ''),
                'sector': info.get('sector', ''),
                'industry': info.get('industry', ''),
                'country': info.get('country', ''),
                'currency': info.get('currency', ''),
                'exchange': info.get('exchange', ''),
                'market_cap': info.get('marketCap', None),
                'pe_ratio': info.get('trailingPE', None),
                'eps': info.get('trailingEps', None),
                'dividend_yield': info.get('dividendYield', None) * 100 if info.get('dividendYield') else None,
                'fifty_two_week_high': info.get('fiftyTwoWeekHigh', None),
                'fifty_two_week_low': info.get('fiftyTwoWeekLow', None),
                'avg_volume': info.get('averageVolume', None),
                'beta': info.get('beta', None),
                'description': info.get('longBusinessSummary', '')
            }
            
            return relevant_info
        except Exception as e:
            raise Exception(f"Erro ao obter informações para {ticker}: {str(e)}")
    
    @staticmethod
    def get_market_summary():
        """
        Obtém um resumo do mercado.
        
        Returns:
            dict: Dicionário com resumo do mercado
        """
        try:
            # Índices principais
            indices = {
                '^BVSP': 'Ibovespa',
                '^DJI': 'Dow Jones',
                '^GSPC': 'S&P 500',
                '^IXIC': 'Nasdaq',
                '^FTSE': 'FTSE 100'
            }
            
            summary = {}
            
            for symbol, name in indices.items():
                data = yf.download(symbol, period='5d')
                
                if not data.empty:
                    last_close = data['Close'].iloc[-1]
                    prev_close = data['Close'].iloc[-2]
                    change = ((last_close / prev_close) - 1) * 100
                    
                    summary[name] = {
                        'symbol': symbol,
                        'last_price': last_close,
                        'change': change,
                        'change_direction': 'up' if change > 0 else 'down' if change < 0 else 'neutral'
                    }
            
            return summary
        except Exception as e:
            raise Exception(f"Erro ao obter resumo do mercado: {str(e)}")
    
    @staticmethod
    def get_sector_performance():
        """
        Obtém o desempenho por setor.
        
        Returns:
            dict: Dicionário com desempenho por setor
        """
        try:
            # ETFs de setores
            sectors = {
                'XLF': 'Financeiro',
                'XLK': 'Tecnologia',
                'XLE': 'Energia',
                'XLV': 'Saúde',
                'XLI': 'Industrial',
                'XLP': 'Consumo Básico',
                'XLY': 'Consumo Discricionário',
                'XLB': 'Materiais',
                'XLU': 'Utilidades',
                'XLRE': 'Imobiliário'
            }
            
            performance = {}
            
            for symbol, name in sectors.items():
                data = yf.download(symbol, period='1mo')
                
                if not data.empty:
                    last_close = data['Close'].iloc[-1]
                    month_start = data['Close'].iloc[0]
                    change = ((last_close / month_start) - 1) * 100
                    
                    performance[name] = {
                        'symbol': symbol,
                        'change': change,
                        'change_direction': 'up' if change > 0 else 'down' if change < 0 else 'neutral'
                    }
            
            # Ordena por desempenho
            performance = dict(sorted(performance.items(), key=lambda x: x[1]['change'], reverse=True))
            
            return performance
        except Exception as e:
            raise Exception(f"Erro ao obter desempenho por setor: {str(e)}")
