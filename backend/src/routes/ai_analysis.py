"""
Sistema de Análise com IA para Mercado Financeiro - Versão Demo

Este módulo implementa uma versão demo das funcionalidades principais de análise com IA
conforme especificado no TCC, usando dados simulados para demonstração.

Autor: Rafael Lima Caires
Data: Junho 2025
Versão: 2.0 - Demo com dados simulados
"""

from flask import Blueprint, request, jsonify
from flask_cors import cross_origin
import traceback
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging

# Configuração de logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cria blueprint para análise com IA
ai_analysis_bp = Blueprint('ai_analysis', __name__)

class AIAnalysisService:
    """
    Serviço principal para análise com IA do sistema financeiro.
    Implementa as funcionalidades descritas no TCC usando dados simulados.
    """
    
    def __init__(self):
        """Inicializa o serviço de análise com IA."""
        self.risk_profiles = {
            'conservador': {
                'max_volatility': 0.15,
                'min_dividend_yield': 0.04,
                'preferred_sectors': ['Financeiro', 'Consumo Básico', 'Utilities'],
                'risk_tolerance': 'baixo'
            },
            'moderado': {
                'max_volatility': 0.25,
                'min_dividend_yield': 0.02,
                'preferred_sectors': ['Financeiro', 'Industrial', 'Materiais'],
                'risk_tolerance': 'médio'
            },
            'agressivo': {
                'max_volatility': 0.40,
                'min_dividend_yield': 0.00,
                'preferred_sectors': ['Tecnologia', 'Consumo Discricionário'],
                'risk_tolerance': 'alto'
            }
        }
        
        # Base de ativos brasileiros para análise (dados simulados)
        self.brazilian_stocks = {
            'PETR4': {'name': 'Petrobras', 'sector': 'Energia', 'price': 25.50},
            'VALE3': {'name': 'Vale', 'sector': 'Materiais', 'price': 65.80},
            'ITUB4': {'name': 'Itaú Unibanco', 'sector': 'Financeiro', 'price': 32.45},
            'BBDC4': {'name': 'Bradesco', 'sector': 'Financeiro', 'price': 28.90},
            'ABEV3': {'name': 'Ambev', 'sector': 'Consumo Básico', 'price': 12.75},
            'WEGE3': {'name': 'WEG', 'sector': 'Industrial', 'price': 45.20},
            'MGLU3': {'name': 'Magazine Luiza', 'sector': 'Consumo Discricionário', 'price': 8.45},
            'RENT3': {'name': 'Localiza', 'sector': 'Consumo Discricionário', 'price': 55.30},
            'LREN3': {'name': 'Lojas Renner', 'sector': 'Consumo Discricionário', 'price': 18.60},
            'SUZB3': {'name': 'Suzano', 'sector': 'Materiais', 'price': 42.15}
        }
    
    def generate_simulated_data(self, ticker, period_days=252):
        """
        Gera dados simulados para demonstração.
        
        Args:
            ticker (str): Código do ativo
            period_days (int): Número de dias de dados históricos
            
        Returns:
            pd.DataFrame: Dados históricos simulados
        """
        try:
            # Obtém preço base do ativo
            base_price = self.brazilian_stocks.get(ticker, {}).get('price', 100.0)
            
            # Gera datas
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days)
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
            
            # Parâmetros para simulação
            np.random.seed(42)  # Para resultados reproduzíveis
            returns = np.random.normal(0.0005, 0.02, len(dates))  # Retornos diários
            
            # Gera preços usando random walk
            prices = [base_price]
            for ret in returns[1:]:
                new_price = prices[-1] * (1 + ret)
                prices.append(max(new_price, 0.01))  # Evita preços negativos
            
            # Gera outros dados OHLCV
            data = []
            for i, (date, price) in enumerate(zip(dates, prices)):
                # Simula variação intraday
                high = price * (1 + abs(np.random.normal(0, 0.01)))
                low = price * (1 - abs(np.random.normal(0, 0.01)))
                open_price = prices[i-1] if i > 0 else price
                volume = int(np.random.normal(1000000, 200000))
                
                data.append({
                    'Date': date,
                    'Open': open_price,
                    'High': high,
                    'Low': low,
                    'Close': price,
                    'Volume': max(volume, 100000)
                })
            
            df = pd.DataFrame(data)
            df.set_index('Date', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Erro ao gerar dados simulados para {ticker}: {e}")
            raise
    
    def calculate_technical_indicators(self, data):
        """
        Calcula indicadores técnicos para análise.
        
        Args:
            data (pd.DataFrame): Dados históricos do ativo
            
        Returns:
            dict: Indicadores técnicos calculados
        """
        try:
            # Preço atual
            current_price = data['Close'].iloc[-1]
            
            # Médias móveis
            sma_20 = data['Close'].rolling(window=20).mean().iloc[-1]
            sma_50 = data['Close'].rolling(window=50).mean().iloc[-1]
            sma_200 = data['Close'].rolling(window=200).mean().iloc[-1] if len(data) >= 200 else None
            
            # RSI (Relative Strength Index)
            delta = data['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = 100 - (100 / (1 + rs)).iloc[-1]
            
            # MACD
            ema_12 = data['Close'].ewm(span=12).mean()
            ema_26 = data['Close'].ewm(span=26).mean()
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_current = macd.iloc[-1]
            macd_signal_current = macd_signal.iloc[-1]
            
            # Bollinger Bands
            bb_middle = data['Close'].rolling(window=20).mean()
            bb_std = data['Close'].rolling(window=20).std()
            bb_upper = (bb_middle + (bb_std * 2)).iloc[-1]
            bb_lower = (bb_middle - (bb_std * 2)).iloc[-1]
            
            # Volatilidade
            returns = data['Close'].pct_change().dropna()
            volatility = returns.std() * np.sqrt(252)  # Volatilidade anualizada
            
            return {
                'current_price': float(current_price),
                'sma_20': float(sma_20),
                'sma_50': float(sma_50),
                'sma_200': float(sma_200) if sma_200 is not None else None,
                'rsi': float(rsi),
                'macd': float(macd_current),
                'macd_signal': float(macd_signal_current),
                'bb_upper': float(bb_upper),
                'bb_lower': float(bb_lower),
                'volatility': float(volatility),
                'volume_avg': float(data['Volume'].tail(20).mean())
            }
            
        except Exception as e:
            logger.error(f"Erro ao calcular indicadores técnicos: {e}")
            raise
    
    def simple_lstm_prediction(self, data, days_ahead=30):
        """
        Implementação simplificada de previsão LSTM usando dados simulados.
        
        Args:
            data (pd.DataFrame): Dados históricos
            days_ahead (int): Número de dias para prever
            
        Returns:
            dict: Previsões geradas
        """
        try:
            prices = data['Close'].values
            returns = data['Close'].pct_change().dropna()
            
            # Calcula tendência recente
            recent_trend = np.mean(returns.tail(10))
            volatility = returns.std()
            
            # Gera previsões baseadas na tendência
            predictions = []
            last_price = prices[-1]
            
            np.random.seed(42)  # Para resultados reproduzíveis
            
            for i in range(days_ahead):
                # Adiciona ruído baseado na volatilidade histórica
                noise = np.random.normal(0, volatility * 0.5)
                
                # Aplica tendência com decaimento
                trend_factor = recent_trend * (0.95 ** i)  # Decaimento da tendência
                
                # Calcula próximo preço
                next_price = last_price * (1 + trend_factor + noise)
                
                # Calcula intervalo de confiança
                confidence_interval = last_price * volatility * 1.96  # 95% de confiança
                
                predictions.append({
                    'day': i + 1,
                    'date': (datetime.now() + timedelta(days=i+1)).strftime('%Y-%m-%d'),
                    'predicted_price': float(next_price),
                    'lower_bound': float(next_price - confidence_interval),
                    'upper_bound': float(next_price + confidence_interval),
                    'confidence': float(max(0.5, 0.9 - (i * 0.02)))  # Confiança decrescente
                })
                
                last_price = next_price
            
            return {
                'model_type': 'LSTM Simplificado (Demo)',
                'base_price': float(prices[-1]),
                'trend_detected': 'alta' if recent_trend > 0 else 'baixa' if recent_trend < 0 else 'lateral',
                'volatility': float(volatility),
                'predictions': predictions
            }
            
        except Exception as e:
            logger.error(f"Erro na previsão LSTM: {e}")
            raise
    
    def analyze_sentiment(self, ticker):
        """
        Análise simplificada de sentimento usando dados simulados.
        
        Args:
            ticker (str): Código do ativo
            
        Returns:
            dict: Análise de sentimento
        """
        try:
            # Simula análise de sentimento baseada no ticker
            np.random.seed(hash(ticker) % 1000)  # Seed baseado no ticker para consistência
            
            sentiment_score = np.random.uniform(0.2, 0.8)
            
            if sentiment_score > 0.6:
                sentiment_label = 'Muito Positivo'
            elif sentiment_score > 0.5:
                sentiment_label = 'Positivo'
            elif sentiment_score > 0.4:
                sentiment_label = 'Neutro'
            elif sentiment_score > 0.3:
                sentiment_label = 'Negativo'
            else:
                sentiment_label = 'Muito Negativo'
            
            recent_performance = np.random.uniform(-5, 5)
            volume_trend = np.random.uniform(0.8, 1.3)
            
            return {
                'ticker': ticker,
                'sentiment_score': float(sentiment_score),
                'sentiment_label': sentiment_label,
                'recent_performance': float(recent_performance),
                'volume_trend': float(volume_trend),
                'analysis_date': datetime.now().strftime('%Y-%m-%d'),
                'confidence': 0.75,
                'methodology': 'Análise simulada para demonstração'
            }
            
        except Exception as e:
            logger.error(f"Erro na análise de sentimento: {e}")
            raise
    
    def calculate_risk_metrics(self, data):
        """
        Calcula métricas de risco para um ativo.
        
        Args:
            data (pd.DataFrame): Dados históricos
            
        Returns:
            dict: Métricas de risco calculadas
        """
        try:
            returns = data['Close'].pct_change().dropna()
            
            # Value at Risk (VaR) - 95% de confiança
            var_95 = np.percentile(returns, 5)
            
            # Conditional VaR (CVaR)
            cvar_95 = returns[returns <= var_95].mean()
            
            # Volatilidade anualizada
            volatility = returns.std() * np.sqrt(252)
            
            # Sharpe Ratio (assumindo taxa livre de risco de 2%)
            risk_free_rate = 0.02 / 252  # Taxa diária
            excess_returns = returns - risk_free_rate
            sharpe_ratio = excess_returns.mean() / returns.std() if returns.std() > 0 else 0
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Beta simulado
            beta = np.random.uniform(0.7, 1.3)
            
            return {
                'var_95': float(var_95 * 100),  # Em percentual
                'cvar_95': float(cvar_95 * 100),
                'volatility': float(volatility * 100),
                'sharpe_ratio': float(sharpe_ratio),
                'max_drawdown': float(max_drawdown * 100),
                'beta': float(beta),
                'risk_classification': self._classify_risk(volatility)
            }
            
        except Exception as e:
            logger.error(f"Erro no cálculo de métricas de risco: {e}")
            raise
    
    def _classify_risk(self, volatility):
        """Classifica o risco baseado na volatilidade."""
        if volatility < 0.15:
            return 'Baixo'
        elif volatility < 0.25:
            return 'Moderado'
        elif volatility < 0.35:
            return 'Alto'
        else:
            return 'Muito Alto'
    
    def generate_recommendations(self, user_profile, portfolio_value=10000):
        """
        Gera recomendações personalizadas baseadas no perfil do usuário.
        
        Args:
            user_profile (dict): Perfil do usuário
            portfolio_value (float): Valor do portfólio
            
        Returns:
            dict: Recomendações personalizadas
        """
        try:
            risk_profile = user_profile.get('risk_profile', 'moderado').lower()
            profile_config = self.risk_profiles.get(risk_profile, self.risk_profiles['moderado'])
            
            recommendations = []
            
            # Analisa cada ativo da base
            for ticker, info in self.brazilian_stocks.items():
                try:
                    # Gera dados simulados
                    data = self.generate_simulated_data(ticker, period_days=180)
                    
                    # Calcula métricas
                    risk_metrics = self.calculate_risk_metrics(data)
                    technical_indicators = self.calculate_technical_indicators(data)
                    sentiment = self.analyze_sentiment(ticker)
                    
                    # Verifica se o ativo se adequa ao perfil
                    if risk_metrics['volatility'] / 100 <= profile_config['max_volatility']:
                        
                        # Calcula score de recomendação
                        score = 0
                        
                        # Score baseado em performance técnica (40%)
                        if technical_indicators['rsi'] < 70 and technical_indicators['rsi'] > 30:
                            score += 0.4 * (1 - abs(technical_indicators['rsi'] - 50) / 50)
                        
                        # Score baseado em sentimento (30%)
                        score += 0.3 * sentiment['sentiment_score']
                        
                        # Score baseado em risco-retorno (30%)
                        if risk_metrics['sharpe_ratio'] > 0:
                            score += 0.3 * min(1, risk_metrics['sharpe_ratio'] / 2)
                        
                        # Bonus para setores preferidos
                        if info['sector'] in profile_config['preferred_sectors']:
                            score += 0.1
                        
                        # Calcula alocação sugerida
                        max_allocation = 0.15 if risk_profile == 'conservador' else 0.20 if risk_profile == 'moderado' else 0.25
                        suggested_allocation = min(max_allocation, score * 0.3)
                        suggested_value = portfolio_value * suggested_allocation
                        
                        recommendations.append({
                            'ticker': ticker,
                            'name': info['name'],
                            'sector': info['sector'],
                            'recommendation_score': float(score),
                            'suggested_allocation': float(suggested_allocation),
                            'suggested_value': float(suggested_value),
                            'current_price': technical_indicators['current_price'],
                            'risk_level': risk_metrics['risk_classification'],
                            'sentiment': sentiment['sentiment_label'],
                            'reasoning': self._generate_reasoning(score, risk_metrics, sentiment, technical_indicators)
                        })
                        
                except Exception as e:
                    logger.warning(f"Erro ao analisar {ticker}: {e}")
                    continue
            
            # Ordena por score e pega os top 5
            recommendations.sort(key=lambda x: x['recommendation_score'], reverse=True)
            top_recommendations = recommendations[:5]
            
            return {
                'user_profile': risk_profile,
                'portfolio_value': portfolio_value,
                'recommendations': top_recommendations,
                'total_suggested_allocation': sum(r['suggested_allocation'] for r in top_recommendations),
                'diversification_score': len(set(r['sector'] for r in top_recommendations)) / 5,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'methodology': 'Sistema híbrido baseado em análise técnica, sentimento e perfil de risco (Demo)'
            }
            
        except Exception as e:
            logger.error(f"Erro na geração de recomendações: {e}")
            raise
    
    def _generate_reasoning(self, score, risk_metrics, sentiment, technical_indicators):
        """Gera explicação para a recomendação."""
        reasons = []
        
        if score > 0.7:
            reasons.append("Alta pontuação geral")
        elif score > 0.5:
            reasons.append("Pontuação moderada")
        else:
            reasons.append("Pontuação baixa")
        
        if sentiment['sentiment_score'] > 0.6:
            reasons.append("sentimento positivo")
        elif sentiment['sentiment_score'] < 0.4:
            reasons.append("sentimento negativo")
        
        if risk_metrics['volatility'] < 20:
            reasons.append("baixa volatilidade")
        elif risk_metrics['volatility'] > 30:
            reasons.append("alta volatilidade")
        
        if 30 < technical_indicators['rsi'] < 70:
            reasons.append("RSI em zona neutra")
        
        return ", ".join(reasons)

# Instância global do serviço
ai_service = AIAnalysisService()

@ai_analysis_bp.route('/api/ai-analysis/predict', methods=['POST'])
@cross_origin()
def predict_stock_price():
    """
    Endpoint para previsão de preços usando IA.
    
    Payload esperado:
    {
        "ticker": "PETR4",
        "days_ahead": 30
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({
                'success': False,
                'error': 'Ticker é obrigatório'
            }), 400
        
        ticker = data['ticker']
        days_ahead = data.get('days_ahead', 30)
        
        # Valida parâmetros
        if days_ahead < 1 or days_ahead > 90:
            return jsonify({
                'success': False,
                'error': 'days_ahead deve estar entre 1 e 90'
            }), 400
        
        # Verifica se o ticker está na base
        if ticker not in ai_service.brazilian_stocks:
            return jsonify({
                'success': False,
                'error': f'Ticker {ticker} não encontrado na base de dados'
            }), 400
        
        # Gera dados simulados
        stock_data = ai_service.generate_simulated_data(ticker, period_days=730)
        
        # Gera previsões
        predictions = ai_service.simple_lstm_prediction(stock_data, days_ahead)
        
        # Calcula indicadores técnicos
        technical_indicators = ai_service.calculate_technical_indicators(stock_data)
        
        return jsonify({
            'success': True,
            'data': {
                'ticker': ticker,
                'predictions': predictions,
                'technical_indicators': technical_indicators,
                'analysis_timestamp': datetime.now().isoformat(),
                'note': 'Dados simulados para demonstração'
            }
        })
        
    except Exception as e:
        logger.error(f"Erro na previsão: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
        }), 500

@ai_analysis_bp.route('/api/ai-analysis/sentiment', methods=['POST'])
@cross_origin()
def analyze_stock_sentiment():
    """
    Endpoint para análise de sentimento.
    
    Payload esperado:
    {
        "ticker": "PETR4"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({
                'success': False,
                'error': 'Ticker é obrigatório'
            }), 400
        
        ticker = data['ticker']
        
        # Verifica se o ticker está na base
        if ticker not in ai_service.brazilian_stocks:
            return jsonify({
                'success': False,
                'error': f'Ticker {ticker} não encontrado na base de dados'
            }), 400
        
        # Analisa sentimento
        sentiment_analysis = ai_service.analyze_sentiment(ticker)
        
        return jsonify({
            'success': True,
            'data': sentiment_analysis
        })
        
    except Exception as e:
        logger.error(f"Erro na análise de sentimento: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_analysis_bp.route('/api/ai-analysis/risk', methods=['POST'])
@cross_origin()
def analyze_stock_risk():
    """
    Endpoint para análise de risco.
    
    Payload esperado:
    {
        "ticker": "PETR4"
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({
                'success': False,
                'error': 'Ticker é obrigatório'
            }), 400
        
        ticker = data['ticker']
        
        # Verifica se o ticker está na base
        if ticker not in ai_service.brazilian_stocks:
            return jsonify({
                'success': False,
                'error': f'Ticker {ticker} não encontrado na base de dados'
            }), 400
        
        # Gera dados simulados
        stock_data = ai_service.generate_simulated_data(ticker, period_days=730)
        
        # Calcula métricas de risco
        risk_metrics = ai_service.calculate_risk_metrics(stock_data)
        
        return jsonify({
            'success': True,
            'data': {
                'ticker': ticker,
                'risk_metrics': risk_metrics,
                'analysis_timestamp': datetime.now().isoformat(),
                'note': 'Dados simulados para demonstração'
            }
        })
        
    except Exception as e:
        logger.error(f"Erro na análise de risco: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_analysis_bp.route('/api/ai-analysis/recommendations', methods=['POST'])
@cross_origin()
def get_investment_recommendations():
    """
    Endpoint para recomendações de investimento.
    
    Payload esperado:
    {
        "user_profile": {
            "risk_profile": "moderado",
            "investment_goals": ["crescimento"],
            "time_horizon": "longo_prazo"
        },
        "portfolio_value": 10000
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'user_profile' not in data:
            return jsonify({
                'success': False,
                'error': 'Perfil do usuário é obrigatório'
            }), 400
        
        user_profile = data['user_profile']
        portfolio_value = data.get('portfolio_value', 10000)
        
        # Valida perfil de risco
        risk_profile = user_profile.get('risk_profile', 'moderado').lower()
        if risk_profile not in ['conservador', 'moderado', 'agressivo']:
            return jsonify({
                'success': False,
                'error': 'Perfil de risco deve ser: conservador, moderado ou agressivo'
            }), 400
        
        # Gera recomendações
        recommendations = ai_service.generate_recommendations(user_profile, portfolio_value)
        
        return jsonify({
            'success': True,
            'data': recommendations
        })
        
    except Exception as e:
        logger.error(f"Erro na geração de recomendações: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_analysis_bp.route('/api/ai-analysis/complete-analysis', methods=['POST'])
@cross_origin()
def complete_stock_analysis():
    """
    Endpoint para análise completa de um ativo.
    
    Payload esperado:
    {
        "ticker": "PETR4",
        "days_ahead": 30
    }
    """
    try:
        data = request.get_json()
        
        if not data or 'ticker' not in data:
            return jsonify({
                'success': False,
                'error': 'Ticker é obrigatório'
            }), 400
        
        ticker = data['ticker']
        days_ahead = data.get('days_ahead', 30)
        
        # Verifica se o ticker está na base
        if ticker not in ai_service.brazilian_stocks:
            return jsonify({
                'success': False,
                'error': f'Ticker {ticker} não encontrado na base de dados. Tickers disponíveis: {list(ai_service.brazilian_stocks.keys())}'
            }), 400
        
        # Gera dados simulados
        stock_data = ai_service.generate_simulated_data(ticker, period_days=730)
        
        # Executa todas as análises
        predictions = ai_service.simple_lstm_prediction(stock_data, days_ahead)
        technical_indicators = ai_service.calculate_technical_indicators(stock_data)
        sentiment_analysis = ai_service.analyze_sentiment(ticker)
        risk_metrics = ai_service.calculate_risk_metrics(stock_data)
        
        # Gera resumo executivo
        current_price = technical_indicators['current_price']
        predicted_price_30d = predictions['predictions'][min(29, len(predictions['predictions'])-1)]['predicted_price']
        expected_return = ((predicted_price_30d / current_price) - 1) * 100
        
        executive_summary = {
            'ticker': ticker,
            'current_price': current_price,
            'predicted_price_30d': predicted_price_30d,
            'expected_return_30d': expected_return,
            'risk_level': risk_metrics['risk_classification'],
            'sentiment': sentiment_analysis['sentiment_label'],
            'recommendation': 'COMPRA' if expected_return > 5 and sentiment_analysis['sentiment_score'] > 0.6 else 
                           'VENDA' if expected_return < -5 and sentiment_analysis['sentiment_score'] < 0.4 else 'MANTER'
        }
        
        return jsonify({
            'success': True,
            'data': {
                'executive_summary': executive_summary,
                'predictions': predictions,
                'technical_indicators': technical_indicators,
                'sentiment_analysis': sentiment_analysis,
                'risk_metrics': risk_metrics,
                'analysis_timestamp': datetime.now().isoformat(),
                'note': 'Análise baseada em dados simulados para demonstração'
            }
        })
        
    except Exception as e:
        logger.error(f"Erro na análise completa: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@ai_analysis_bp.route('/api/ai-analysis/health', methods=['GET'])
@cross_origin()
def health_check():
    """Endpoint para verificar a saúde do serviço de IA."""
    return jsonify({
        'success': True,
        'status': 'healthy',
        'service': 'AI Analysis Service',
        'version': '2.0',
        'timestamp': datetime.now().isoformat(),
        'available_tickers': list(ai_service.brazilian_stocks.keys()),
        'available_endpoints': [
            '/api/ai-analysis/predict',
            '/api/ai-analysis/sentiment',
            '/api/ai-analysis/risk',
            '/api/ai-analysis/recommendations',
            '/api/ai-analysis/complete-analysis'
        ],
        'note': 'Sistema demo usando dados simulados para demonstração'
    })

