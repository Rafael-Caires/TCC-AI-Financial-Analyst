"""
Sistema de análise de sentimentos para mercado financeiro

Este módulo implementa análise de sentimentos de notícias e redes sociais
para auxiliar nas previsões do mercado financeiro.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import pandas as pd
import numpy as np
import requests
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
import yfinance as yf
from bs4 import BeautifulSoup
import time
import logging

# Download necessário para NLTK
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """
    Classe para análise de sentimentos de notícias financeiras e redes sociais.
    """
    
    def __init__(self):
        """
        Inicializa o analisador de sentimentos.
        """
        self.sia = SentimentIntensityAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Palavras-chave financeiras para filtrar notícias relevantes
        self.financial_keywords = [
            'stock', 'market', 'trading', 'investment', 'earnings', 'revenue',
            'profit', 'loss', 'bull', 'bear', 'volatility', 'dividend',
            'ação', 'mercado', 'investimento', 'lucro', 'prejuízo', 'bolsa',
            'alta', 'baixa', 'volatilidade', 'dividendo'
        ]
        
        # Dicionário de sentimentos específicos do mercado financeiro
        self.financial_sentiment_words = {
            'positive': [
                'crescimento', 'alta', 'valorização', 'lucro', 'ganho', 'otimista',
                'bullish', 'rally', 'surge', 'boom', 'profit', 'gain', 'rise',
                'increase', 'growth', 'positive', 'strong', 'robust'
            ],
            'negative': [
                'queda', 'baixa', 'desvalorização', 'prejuízo', 'perda', 'pessimista',
                'bearish', 'crash', 'decline', 'fall', 'loss', 'drop', 'decrease',
                'negative', 'weak', 'poor', 'recession', 'crisis'
            ]
        }
    
    def analyze_text_sentiment(self, text: str) -> Dict[str, float]:
        """
        Analisa o sentimento de um texto usando múltiplas abordagens.
        
        Args:
            text (str): Texto para análise
            
        Returns:
            Dict[str, float]: Scores de sentimento
        """
        if not text or not isinstance(text, str):
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'textblob_polarity': 0.0,
                'financial_sentiment': 0.0
            }
        
        # Limpa o texto
        cleaned_text = self._clean_text(text)
        
        # VADER sentiment
        vader_scores = self.sia.polarity_scores(cleaned_text)
        
        # TextBlob sentiment
        blob = TextBlob(cleaned_text)
        textblob_polarity = blob.sentiment.polarity
        
        # Sentimento financeiro customizado
        financial_sentiment = self._calculate_financial_sentiment(cleaned_text)
        
        return {
            'compound': vader_scores['compound'],
            'positive': vader_scores['pos'],
            'negative': vader_scores['neg'],
            'neutral': vader_scores['neu'],
            'textblob_polarity': textblob_polarity,
            'financial_sentiment': financial_sentiment
        }
    
    def _clean_text(self, text: str) -> str:
        """
        Limpa e preprocessa o texto.
        
        Args:
            text (str): Texto original
            
        Returns:
            str: Texto limpo
        """
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
        
        # Remove menções e hashtags (mantém o conteúdo)
        text = re.sub(r'@\w+|#\w+', '', text)
        
        # Remove caracteres especiais, mantém pontuação básica
        text = re.sub(r'[^\w\s.,!?-]', '', text)
        
        # Remove espaços extras
        text = ' '.join(text.split())
        
        return text.lower()
    
    def _calculate_financial_sentiment(self, text: str) -> float:
        """
        Calcula sentimento baseado em palavras-chave financeiras.
        
        Args:
            text (str): Texto para análise
            
        Returns:
            float: Score de sentimento financeiro (-1 a 1)
        """
        words = text.lower().split()
        
        positive_count = 0
        negative_count = 0
        
        for word in words:
            if word in self.financial_sentiment_words['positive']:
                positive_count += 1
            elif word in self.financial_sentiment_words['negative']:
                negative_count += 1
        
        total_financial_words = positive_count + negative_count
        
        if total_financial_words == 0:
            return 0.0
        
        return (positive_count - negative_count) / total_financial_words
    
    def get_news_sentiment(self, ticker: str, days_back: int = 7) -> Dict[str, Any]:
        """
        Obtém sentimento de notícias para um ticker específico.
        
        Args:
            ticker (str): Símbolo do ativo
            days_back (int): Número de dias para buscar notícias
            
        Returns:
            Dict[str, Any]: Análise de sentimento das notícias
        """
        try:
            # Busca notícias usando yfinance
            stock = yf.Ticker(ticker)
            news = stock.news
            
            if not news:
                return self._empty_sentiment_result()
            
            # Analisa sentimento de cada notícia
            sentiments = []
            news_data = []
            
            cutoff_date = datetime.now() - timedelta(days=days_back)
            
            for article in news:
                # Verifica se a notícia é recente
                article_date = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                
                if article_date < cutoff_date:
                    continue
                
                title = article.get('title', '')
                summary = article.get('summary', '')
                
                # Combina título e resumo
                full_text = f"{title}. {summary}"
                
                # Analisa sentimento
                sentiment = self.analyze_text_sentiment(full_text)
                sentiments.append(sentiment)
                
                news_data.append({
                    'title': title,
                    'summary': summary,
                    'date': article_date.strftime('%Y-%m-%d %H:%M:%S'),
                    'url': article.get('link', ''),
                    'sentiment': sentiment
                })
            
            if not sentiments:
                return self._empty_sentiment_result()
            
            # Calcula métricas agregadas
            avg_sentiment = self._calculate_average_sentiment(sentiments)
            
            return {
                'ticker': ticker,
                'period': f'{days_back} days',
                'news_count': len(sentiments),
                'average_sentiment': avg_sentiment,
                'sentiment_trend': self._calculate_sentiment_trend(sentiments, news_data),
                'news_details': news_data[:10],  # Limita a 10 notícias mais recentes
                'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Erro ao obter sentimento de notícias para {ticker}: {e}")
            return self._empty_sentiment_result()
    
    def _empty_sentiment_result(self) -> Dict[str, Any]:
        """
        Retorna resultado vazio para casos de erro.
        
        Returns:
            Dict[str, Any]: Resultado vazio
        """
        return {
            'ticker': '',
            'period': '',
            'news_count': 0,
            'average_sentiment': {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'textblob_polarity': 0.0,
                'financial_sentiment': 0.0
            },
            'sentiment_trend': 'neutral',
            'news_details': [],
            'last_updated': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def _calculate_average_sentiment(self, sentiments: List[Dict[str, float]]) -> Dict[str, float]:
        """
        Calcula sentimento médio de uma lista de sentimentos.
        
        Args:
            sentiments (List[Dict[str, float]]): Lista de sentimentos
            
        Returns:
            Dict[str, float]: Sentimento médio
        """
        if not sentiments:
            return {
                'compound': 0.0,
                'positive': 0.0,
                'negative': 0.0,
                'neutral': 1.0,
                'textblob_polarity': 0.0,
                'financial_sentiment': 0.0
            }
        
        avg_sentiment = {}
        for key in sentiments[0].keys():
            avg_sentiment[key] = np.mean([s[key] for s in sentiments])
        
        return avg_sentiment
    
    def _calculate_sentiment_trend(self, sentiments: List[Dict[str, float]], 
                                 news_data: List[Dict[str, Any]]) -> str:
        """
        Calcula tendência do sentimento ao longo do tempo.
        
        Args:
            sentiments (List[Dict[str, float]]): Lista de sentimentos
            news_data (List[Dict[str, Any]]): Dados das notícias
            
        Returns:
            str: Tendência ('improving', 'declining', 'stable', 'neutral')
        """
        if len(sentiments) < 2:
            return 'neutral'
        
        # Ordena por data
        combined = list(zip(sentiments, news_data))
        combined.sort(key=lambda x: x[1]['date'])
        
        # Divide em duas metades
        mid_point = len(combined) // 2
        first_half = [item[0] for item in combined[:mid_point]]
        second_half = [item[0] for item in combined[mid_point:]]
        
        # Calcula sentimento médio de cada metade
        first_avg = np.mean([s['compound'] for s in first_half])
        second_avg = np.mean([s['compound'] for s in second_half])
        
        # Determina tendência
        diff = second_avg - first_avg
        
        if diff > 0.1:
            return 'improving'
        elif diff < -0.1:
            return 'declining'
        else:
            return 'stable'
    
    def get_market_sentiment_summary(self, tickers: List[str]) -> Dict[str, Any]:
        """
        Obtém resumo de sentimento para múltiplos tickers.
        
        Args:
            tickers (List[str]): Lista de símbolos
            
        Returns:
            Dict[str, Any]: Resumo de sentimento do mercado
        """
        ticker_sentiments = {}
        all_sentiments = []
        
        for ticker in tickers:
            sentiment_data = self.get_news_sentiment(ticker, days_back=3)
            ticker_sentiments[ticker] = sentiment_data
            
            if sentiment_data['news_count'] > 0:
                all_sentiments.append(sentiment_data['average_sentiment'])
            
            # Delay para evitar rate limiting
            time.sleep(0.5)
        
        # Calcula sentimento geral do mercado
        if all_sentiments:
            market_sentiment = self._calculate_average_sentiment(all_sentiments)
            
            # Classifica sentimento geral
            compound_score = market_sentiment['compound']
            if compound_score >= 0.2:
                market_mood = 'bullish'
            elif compound_score <= -0.2:
                market_mood = 'bearish'
            else:
                market_mood = 'neutral'
        else:
            market_sentiment = self._empty_sentiment_result()['average_sentiment']
            market_mood = 'neutral'
        
        return {
            'market_mood': market_mood,
            'overall_sentiment': market_sentiment,
            'ticker_sentiments': ticker_sentiments,
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'tickers_analyzed': len(tickers),
            'tickers_with_news': len([t for t in ticker_sentiments.values() if t['news_count'] > 0])
        }
    
    def get_sentiment_signal(self, ticker: str) -> Dict[str, Any]:
        """
        Gera sinal de trading baseado em sentimento.
        
        Args:
            ticker (str): Símbolo do ativo
            
        Returns:
            Dict[str, Any]: Sinal de trading
        """
        sentiment_data = self.get_news_sentiment(ticker, days_back=5)
        
        if sentiment_data['news_count'] == 0:
            return {
                'signal': 'hold',
                'confidence': 0.0,
                'reason': 'Insufficient news data',
                'sentiment_score': 0.0
            }
        
        avg_sentiment = sentiment_data['average_sentiment']
        compound_score = avg_sentiment['compound']
        financial_score = avg_sentiment['financial_sentiment']
        
        # Combina scores
        combined_score = (compound_score * 0.6) + (financial_score * 0.4)
        
        # Gera sinal
        if combined_score >= 0.3:
            signal = 'buy'
            confidence = min(0.9, abs(combined_score))
        elif combined_score <= -0.3:
            signal = 'sell'
            confidence = min(0.9, abs(combined_score))
        else:
            signal = 'hold'
            confidence = 0.5
        
        # Ajusta confiança baseada na quantidade de notícias
        news_factor = min(1.0, sentiment_data['news_count'] / 10)
        confidence *= news_factor
        
        return {
            'signal': signal,
            'confidence': float(confidence),
            'reason': self._generate_signal_reason(signal, combined_score, sentiment_data),
            'sentiment_score': float(combined_score),
            'news_count': sentiment_data['news_count'],
            'sentiment_trend': sentiment_data['sentiment_trend']
        }
    
    def _generate_signal_reason(self, signal: str, score: float, 
                              sentiment_data: Dict[str, Any]) -> str:
        """
        Gera explicação para o sinal de trading.
        
        Args:
            signal (str): Sinal gerado
            score (float): Score de sentimento
            sentiment_data (Dict[str, Any]): Dados de sentimento
            
        Returns:
            str: Explicação do sinal
        """
        trend = sentiment_data['sentiment_trend']
        news_count = sentiment_data['news_count']
        
        if signal == 'buy':
            return f"Sentimento positivo (score: {score:.2f}) com {news_count} notícias. Tendência: {trend}"
        elif signal == 'sell':
            return f"Sentimento negativo (score: {score:.2f}) com {news_count} notícias. Tendência: {trend}"
        else:
            return f"Sentimento neutro (score: {score:.2f}) com {news_count} notícias. Tendência: {trend}"
    
    def analyze_earnings_sentiment(self, ticker: str, earnings_date: str) -> Dict[str, Any]:
        """
        Analisa sentimento específico para período de earnings.
        
        Args:
            ticker (str): Símbolo do ativo
            earnings_date (str): Data dos earnings (YYYY-MM-DD)
            
        Returns:
            Dict[str, Any]: Análise de sentimento para earnings
        """
        try:
            earnings_dt = datetime.strptime(earnings_date, '%Y-%m-%d')
            
            # Analisa sentimento antes e depois dos earnings
            before_date = earnings_dt - timedelta(days=3)
            after_date = earnings_dt + timedelta(days=3)
            
            # Busca notícias em períodos específicos
            before_sentiment = self._get_sentiment_for_period(ticker, before_date, earnings_dt)
            after_sentiment = self._get_sentiment_for_period(ticker, earnings_dt, after_date)
            
            # Calcula mudança no sentimento
            sentiment_change = (after_sentiment['compound'] - before_sentiment['compound'])
            
            return {
                'ticker': ticker,
                'earnings_date': earnings_date,
                'pre_earnings_sentiment': before_sentiment,
                'post_earnings_sentiment': after_sentiment,
                'sentiment_change': float(sentiment_change),
                'earnings_reaction': self._classify_earnings_reaction(sentiment_change),
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            self.logger.error(f"Erro na análise de earnings para {ticker}: {e}")
            return {
                'ticker': ticker,
                'earnings_date': earnings_date,
                'error': str(e)
            }
    
    def _get_sentiment_for_period(self, ticker: str, start_date: datetime, 
                                end_date: datetime) -> Dict[str, float]:
        """
        Obtém sentimento para um período específico.
        
        Args:
            ticker (str): Símbolo do ativo
            start_date (datetime): Data inicial
            end_date (datetime): Data final
            
        Returns:
            Dict[str, float]: Sentimento médio do período
        """
        try:
            stock = yf.Ticker(ticker)
            news = stock.news
            
            period_sentiments = []
            
            for article in news:
                article_date = datetime.fromtimestamp(article.get('providerPublishTime', 0))
                
                if start_date <= article_date <= end_date:
                    title = article.get('title', '')
                    summary = article.get('summary', '')
                    full_text = f"{title}. {summary}"
                    
                    sentiment = self.analyze_text_sentiment(full_text)
                    period_sentiments.append(sentiment)
            
            if period_sentiments:
                return self._calculate_average_sentiment(period_sentiments)
            else:
                return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
                
        except Exception:
            return {'compound': 0.0, 'positive': 0.0, 'negative': 0.0, 'neutral': 1.0}
    
    def _classify_earnings_reaction(self, sentiment_change: float) -> str:
        """
        Classifica a reação aos earnings baseada na mudança de sentimento.
        
        Args:
            sentiment_change (float): Mudança no sentimento
            
        Returns:
            str: Classificação da reação
        """
        if sentiment_change >= 0.3:
            return 'very_positive'
        elif sentiment_change >= 0.1:
            return 'positive'
        elif sentiment_change <= -0.3:
            return 'very_negative'
        elif sentiment_change <= -0.1:
            return 'negative'
        else:
            return 'neutral'

