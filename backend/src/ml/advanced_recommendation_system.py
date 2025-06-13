"""
Sistema avançado de recomendação de investimentos

Este módulo implementa um sistema híbrido de recomendação que combina
filtragem colaborativa, content-based filtering e análise quantitativa.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .risk_analyzer import RiskAnalyzer
from .sentiment_analyzer import SentimentAnalyzer

class AdvancedRecommendationSystem:
    """
    Sistema avançado de recomendação de investimentos que combina múltiplas abordagens.
    """
    
    def __init__(self):
        """
        Inicializa o sistema de recomendação.
        """
        self.risk_analyzer = RiskAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.scaler = StandardScaler()
        
        # Base de dados de ativos (seria expandida em produção)
        self.asset_universe = {
            # Ações brasileiras
            'PETR4.SA': {'sector': 'Energy', 'market_cap': 'Large', 'dividend_yield': 0.08},
            'VALE3.SA': {'sector': 'Materials', 'market_cap': 'Large', 'dividend_yield': 0.12},
            'ITUB4.SA': {'sector': 'Financial', 'market_cap': 'Large', 'dividend_yield': 0.06},
            'BBDC4.SA': {'sector': 'Financial', 'market_cap': 'Large', 'dividend_yield': 0.05},
            'ABEV3.SA': {'sector': 'Consumer Staples', 'market_cap': 'Large', 'dividend_yield': 0.04},
            'WEGE3.SA': {'sector': 'Industrials', 'market_cap': 'Large', 'dividend_yield': 0.02},
            'MGLU3.SA': {'sector': 'Consumer Discretionary', 'market_cap': 'Large', 'dividend_yield': 0.00},
            'RENT3.SA': {'sector': 'Consumer Discretionary', 'market_cap': 'Large', 'dividend_yield': 0.01},
            'LREN3.SA': {'sector': 'Consumer Discretionary', 'market_cap': 'Large', 'dividend_yield': 0.02},
            'SUZB3.SA': {'sector': 'Materials', 'market_cap': 'Large', 'dividend_yield': 0.03},
            
            # FIIs
            'HGLG11.SA': {'sector': 'Real Estate', 'market_cap': 'Medium', 'dividend_yield': 0.10},
            'XPML11.SA': {'sector': 'Real Estate', 'market_cap': 'Medium', 'dividend_yield': 0.09},
            'VISC11.SA': {'sector': 'Real Estate', 'market_cap': 'Medium', 'dividend_yield': 0.08},
            'BCFF11.SA': {'sector': 'Real Estate', 'market_cap': 'Medium', 'dividend_yield': 0.11},
            'KNRI11.SA': {'sector': 'Real Estate', 'market_cap': 'Medium', 'dividend_yield': 0.09},
        }
        
        # Perfis de risco predefinidos
        self.risk_profiles = {
            'conservador': {
                'max_volatility': 0.15,
                'min_dividend_yield': 0.04,
                'max_single_position': 0.15,
                'preferred_sectors': ['Financial', 'Consumer Staples', 'Real Estate'],
                'avoid_sectors': ['Technology', 'Biotechnology']
            },
            'moderado': {
                'max_volatility': 0.25,
                'min_dividend_yield': 0.02,
                'max_single_position': 0.20,
                'preferred_sectors': ['Financial', 'Consumer Staples', 'Industrials', 'Materials'],
                'avoid_sectors': ['Biotechnology']
            },
            'agressivo': {
                'max_volatility': 0.40,
                'min_dividend_yield': 0.00,
                'max_single_position': 0.30,
                'preferred_sectors': ['Technology', 'Consumer Discretionary', 'Energy'],
                'avoid_sectors': []
            }
        }
        
        # Cache para dados de mercado
        self.market_data_cache = {}
        self.cache_timestamp = {}
        
    def get_user_recommendations(self, user_profile: Dict[str, Any], 
                               portfolio_value: float = 10000,
                               num_recommendations: int = 10) -> Dict[str, Any]:
        """
        Gera recomendações personalizadas para um usuário.
        
        Args:
            user_profile (Dict[str, Any]): Perfil do usuário
            portfolio_value (float): Valor do portfólio
            num_recommendations (int): Número de recomendações
            
        Returns:
            Dict[str, Any]: Recomendações personalizadas
        """
        try:
            # Extrai informações do perfil
            risk_profile = user_profile.get('risk_profile', 'moderado').lower()
            current_portfolio = user_profile.get('current_portfolio', {})
            investment_goals = user_profile.get('investment_goals', [])
            time_horizon = user_profile.get('time_horizon', 'medium')  # short, medium, long
            
            # Obtém dados de mercado
            market_data = self._get_market_data()
            
            if market_data.empty:
                return self._empty_recommendations()
            
            # Calcula features dos ativos
            asset_features = self._calculate_asset_features(market_data)
            
            # Filtragem baseada no perfil de risco
            filtered_assets = self._filter_by_risk_profile(asset_features, risk_profile)
            
            # Análise de sentimento
            sentiment_scores = self._get_sentiment_scores(list(filtered_assets.keys()))
            
            # Combina scores
            combined_scores = self._combine_recommendation_scores(
                filtered_assets, sentiment_scores, user_profile
            )
            
            # Seleciona top recomendações
            top_recommendations = self._select_top_recommendations(
                combined_scores, current_portfolio, num_recommendations
            )
            
            # Gera alocação sugerida
            suggested_allocation = self._generate_portfolio_allocation(
                top_recommendations, portfolio_value, risk_profile
            )
            
            # Análise de diversificação
            diversification_analysis = self._analyze_diversification(suggested_allocation)
            
            return {
                'recommendations': top_recommendations,
                'suggested_allocation': suggested_allocation,
                'diversification_analysis': diversification_analysis,
                'risk_profile_used': risk_profile,
                'total_portfolio_value': portfolio_value,
                'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'methodology': 'hybrid_collaborative_content_sentiment'
            }
            
        except Exception as e:
            print(f"Erro ao gerar recomendações: {e}")
            return self._empty_recommendations()
    
    def _get_market_data(self, period: str = '1y') -> pd.DataFrame:
        """
        Obtém dados de mercado para todos os ativos.
        
        Args:
            period (str): Período dos dados
            
        Returns:
            pd.DataFrame: Dados de mercado
        """
        cache_key = f"market_data_{period}"
        current_time = datetime.now()
        
        # Verifica cache (válido por 1 hora)
        if (cache_key in self.market_data_cache and 
            cache_key in self.cache_timestamp and
            (current_time - self.cache_timestamp[cache_key]).seconds < 3600):
            return self.market_data_cache[cache_key]
        
        try:
            tickers = list(self.asset_universe.keys())
            data = yf.download(tickers, period=period, group_by='ticker')
            
            # Reorganiza dados
            market_data = {}
            for ticker in tickers:
                try:
                    if len(tickers) == 1:
                        ticker_data = data
                    else:
                        ticker_data = data[ticker]
                    
                    if not ticker_data.empty:
                        market_data[ticker] = ticker_data['Adj Close'].dropna()
                except:
                    continue
            
            result = pd.DataFrame(market_data)
            
            # Atualiza cache
            self.market_data_cache[cache_key] = result
            self.cache_timestamp[cache_key] = current_time
            
            return result
            
        except Exception as e:
            print(f"Erro ao obter dados de mercado: {e}")
            return pd.DataFrame()
    
    def _calculate_asset_features(self, market_data: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """
        Calcula features quantitativas dos ativos.
        
        Args:
            market_data (pd.DataFrame): Dados de mercado
            
        Returns:
            Dict[str, Dict[str, float]]: Features dos ativos
        """
        asset_features = {}
        
        for ticker in market_data.columns:
            try:
                prices = market_data[ticker].dropna()
                
                if len(prices) < 30:  # Mínimo de dados
                    continue
                
                returns = prices.pct_change().dropna()
                
                # Métricas básicas
                volatility = returns.std() * np.sqrt(252)  # Anualizada
                annual_return = (prices.iloc[-1] / prices.iloc[0]) ** (252 / len(prices)) - 1
                sharpe_ratio = self.risk_analyzer.calculate_sharpe_ratio(returns)
                max_drawdown = self.risk_analyzer.calculate_maximum_drawdown(prices)
                
                # Métricas técnicas
                sma_20 = prices.rolling(20).mean().iloc[-1]
                sma_50 = prices.rolling(50).mean().iloc[-1]
                current_price = prices.iloc[-1]
                
                # Momentum
                momentum_1m = (current_price / prices.iloc[-21]) - 1 if len(prices) > 21 else 0
                momentum_3m = (current_price / prices.iloc[-63]) - 1 if len(prices) > 63 else 0
                
                # RSI simplificado
                delta = returns
                gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
                loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
                rs = gain / loss
                rsi = 100 - (100 / (1 + rs)).iloc[-1] if not rs.iloc[-1] == 0 else 50
                
                # Features do ativo
                asset_info = self.asset_universe.get(ticker, {})
                
                asset_features[ticker] = {
                    'annual_return': float(annual_return),
                    'volatility': float(volatility),
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown['max_drawdown']),
                    'momentum_1m': float(momentum_1m),
                    'momentum_3m': float(momentum_3m),
                    'rsi': float(rsi),
                    'price_vs_sma20': float((current_price / sma_20) - 1),
                    'price_vs_sma50': float((current_price / sma_50) - 1),
                    'dividend_yield': float(asset_info.get('dividend_yield', 0)),
                    'sector': asset_info.get('sector', 'Unknown'),
                    'market_cap': asset_info.get('market_cap', 'Unknown')
                }
                
            except Exception as e:
                print(f"Erro ao calcular features para {ticker}: {e}")
                continue
        
        return asset_features
    
    def _filter_by_risk_profile(self, asset_features: Dict[str, Dict[str, float]], 
                               risk_profile: str) -> Dict[str, Dict[str, float]]:
        """
        Filtra ativos baseado no perfil de risco.
        
        Args:
            asset_features (Dict[str, Dict[str, float]]): Features dos ativos
            risk_profile (str): Perfil de risco
            
        Returns:
            Dict[str, Dict[str, float]]: Ativos filtrados
        """
        profile_config = self.risk_profiles.get(risk_profile, self.risk_profiles['moderado'])
        filtered_assets = {}
        
        for ticker, features in asset_features.items():
            # Filtro de volatilidade
            if features['volatility'] > profile_config['max_volatility']:
                continue
            
            # Filtro de dividend yield
            if features['dividend_yield'] < profile_config['min_dividend_yield']:
                continue
            
            # Filtro de setor
            if features['sector'] in profile_config['avoid_sectors']:
                continue
            
            # Bonus para setores preferidos
            if features['sector'] in profile_config['preferred_sectors']:
                features['sector_bonus'] = 0.1
            else:
                features['sector_bonus'] = 0.0
            
            filtered_assets[ticker] = features
        
        return filtered_assets
    
    def _get_sentiment_scores(self, tickers: List[str]) -> Dict[str, float]:
        """
        Obtém scores de sentimento para os ativos.
        
        Args:
            tickers (List[str]): Lista de tickers
            
        Returns:
            Dict[str, float]: Scores de sentimento
        """
        sentiment_scores = {}
        
        for ticker in tickers:
            try:
                # Remove sufixo .SA para busca de notícias
                clean_ticker = ticker.replace('.SA', '')
                
                sentiment_data = self.sentiment_analyzer.get_news_sentiment(clean_ticker, days_back=3)
                
                if sentiment_data['news_count'] > 0:
                    # Combina diferentes métricas de sentimento
                    compound = sentiment_data['average_sentiment']['compound']
                    financial = sentiment_data['average_sentiment']['financial_sentiment']
                    
                    # Score combinado
                    combined_score = (compound * 0.7) + (financial * 0.3)
                    sentiment_scores[ticker] = combined_score
                else:
                    sentiment_scores[ticker] = 0.0
                    
            except Exception as e:
                print(f"Erro ao obter sentimento para {ticker}: {e}")
                sentiment_scores[ticker] = 0.0
        
        return sentiment_scores
    
    def _combine_recommendation_scores(self, asset_features: Dict[str, Dict[str, float]],
                                     sentiment_scores: Dict[str, float],
                                     user_profile: Dict[str, Any]) -> Dict[str, float]:
        """
        Combina diferentes scores para gerar recomendação final.
        
        Args:
            asset_features (Dict[str, Dict[str, float]]): Features dos ativos
            sentiment_scores (Dict[str, float]): Scores de sentimento
            user_profile (Dict[str, Any]): Perfil do usuário
            
        Returns:
            Dict[str, float]: Scores combinados
        """
        combined_scores = {}
        
        # Normaliza features para scoring
        all_features = pd.DataFrame(asset_features).T
        
        for ticker, features in asset_features.items():
            score = 0.0
            
            # Score de performance (30%)
            performance_score = (
                features['annual_return'] * 0.4 +
                features['sharpe_ratio'] * 0.3 +
                (1 - abs(features['max_drawdown'])) * 0.3
            )
            score += performance_score * 0.3
            
            # Score técnico (25%)
            technical_score = (
                (features['momentum_1m'] + 1) * 0.3 +
                (features['momentum_3m'] + 1) * 0.3 +
                (1 - abs(features['rsi'] - 50) / 50) * 0.2 +  # RSI próximo de 50 é melhor
                features['price_vs_sma20'] * 0.2
            )
            score += technical_score * 0.25
            
            # Score de dividendos (20%)
            dividend_score = features['dividend_yield'] * 10  # Normaliza para 0-1
            score += dividend_score * 0.2
            
            # Score de sentimento (15%)
            sentiment_score = sentiment_scores.get(ticker, 0.0)
            score += (sentiment_score + 1) / 2 * 0.15  # Normaliza de -1,1 para 0,1
            
            # Bonus de setor (10%)
            score += features.get('sector_bonus', 0) * 0.1
            
            combined_scores[ticker] = score
        
        return combined_scores
    
    def _select_top_recommendations(self, combined_scores: Dict[str, float],
                                  current_portfolio: Dict[str, float],
                                  num_recommendations: int) -> List[Dict[str, Any]]:
        """
        Seleciona as top recomendações.
        
        Args:
            combined_scores (Dict[str, float]): Scores combinados
            current_portfolio (Dict[str, float]): Portfólio atual
            num_recommendations (int): Número de recomendações
            
        Returns:
            List[Dict[str, Any]]: Top recomendações
        """
        # Ordena por score
        sorted_assets = sorted(combined_scores.items(), key=lambda x: x[1], reverse=True)
        
        recommendations = []
        
        for ticker, score in sorted_assets[:num_recommendations * 2]:  # Pega mais para filtrar
            try:
                # Obtém dados adicionais
                asset_info = self.asset_universe.get(ticker, {})
                
                # Verifica se já está no portfólio
                already_owned = ticker in current_portfolio
                current_weight = current_portfolio.get(ticker, 0.0)
                
                # Gera sinal de recomendação
                if score > 0.6:
                    recommendation_type = 'strong_buy'
                elif score > 0.4:
                    recommendation_type = 'buy'
                elif score > 0.2:
                    recommendation_type = 'hold'
                else:
                    recommendation_type = 'avoid'
                
                # Só recomenda buy/strong_buy para novos ativos
                if already_owned and recommendation_type in ['buy', 'strong_buy']:
                    if current_weight < 0.15:  # Pode aumentar posição se peso for baixo
                        recommendation_type = 'increase_position'
                    else:
                        recommendation_type = 'hold'
                
                recommendation = {
                    'ticker': ticker,
                    'score': float(score),
                    'recommendation_type': recommendation_type,
                    'sector': asset_info.get('sector', 'Unknown'),
                    'dividend_yield': asset_info.get('dividend_yield', 0),
                    'already_owned': already_owned,
                    'current_weight': float(current_weight),
                    'rationale': self._generate_recommendation_rationale(ticker, score, asset_info)
                }
                
                recommendations.append(recommendation)
                
                if len(recommendations) >= num_recommendations:
                    break
                    
            except Exception as e:
                print(f"Erro ao processar recomendação para {ticker}: {e}")
                continue
        
        return recommendations
    
    def _generate_recommendation_rationale(self, ticker: str, score: float, 
                                         asset_info: Dict[str, Any]) -> str:
        """
        Gera justificativa para a recomendação.
        
        Args:
            ticker (str): Ticker do ativo
            score (float): Score da recomendação
            asset_info (Dict[str, Any]): Informações do ativo
            
        Returns:
            str: Justificativa
        """
        sector = asset_info.get('sector', 'Unknown')
        dividend_yield = asset_info.get('dividend_yield', 0)
        
        rationale_parts = []
        
        if score > 0.6:
            rationale_parts.append("Excelente combinação de performance e fundamentos")
        elif score > 0.4:
            rationale_parts.append("Bons fundamentos e perspectivas positivas")
        else:
            rationale_parts.append("Fundamentos adequados para o perfil")
        
        if dividend_yield > 0.05:
            rationale_parts.append(f"Alto dividend yield ({dividend_yield:.1%})")
        
        rationale_parts.append(f"Setor: {sector}")
        
        return ". ".join(rationale_parts)
    
    def _generate_portfolio_allocation(self, recommendations: List[Dict[str, Any]],
                                     portfolio_value: float,
                                     risk_profile: str) -> Dict[str, Any]:
        """
        Gera alocação sugerida do portfólio.
        
        Args:
            recommendations (List[Dict[str, Any]]): Recomendações
            portfolio_value (float): Valor do portfólio
            risk_profile (str): Perfil de risco
            
        Returns:
            Dict[str, Any]: Alocação sugerida
        """
        profile_config = self.risk_profiles.get(risk_profile, self.risk_profiles['moderado'])
        max_single_position = profile_config['max_single_position']
        
        # Filtra apenas recomendações de compra
        buy_recommendations = [r for r in recommendations 
                             if r['recommendation_type'] in ['buy', 'strong_buy']]
        
        if not buy_recommendations:
            return {'allocations': [], 'total_allocated': 0.0}
        
        # Calcula pesos baseados nos scores
        total_score = sum(r['score'] for r in buy_recommendations)
        
        allocations = []
        total_allocated = 0.0
        
        for rec in buy_recommendations:
            # Peso baseado no score
            base_weight = rec['score'] / total_score
            
            # Aplica limite máximo por posição
            weight = min(base_weight, max_single_position)
            
            # Ajusta para diversificação por setor
            sector_count = len([r for r in buy_recommendations if r['sector'] == rec['sector']])
            if sector_count > 1:
                weight *= 0.8  # Reduz peso se há muitos ativos do mesmo setor
            
            allocation_value = weight * portfolio_value
            
            allocations.append({
                'ticker': rec['ticker'],
                'weight': float(weight),
                'value': float(allocation_value),
                'sector': rec['sector'],
                'recommendation_type': rec['recommendation_type']
            })
            
            total_allocated += weight
        
        # Normaliza pesos se total > 1
        if total_allocated > 1.0:
            for allocation in allocations:
                allocation['weight'] /= total_allocated
                allocation['value'] = allocation['weight'] * portfolio_value
            total_allocated = 1.0
        
        return {
            'allocations': allocations,
            'total_allocated': float(total_allocated),
            'cash_position': float(1.0 - total_allocated),
            'number_of_positions': len(allocations)
        }
    
    def _analyze_diversification(self, allocation: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analisa diversificação da alocação sugerida.
        
        Args:
            allocation (Dict[str, Any]): Alocação sugerida
            
        Returns:
            Dict[str, Any]: Análise de diversificação
        """
        if not allocation['allocations']:
            return {'sector_distribution': {}, 'diversification_score': 0.0}
        
        # Distribuição por setor
        sector_weights = {}
        for alloc in allocation['allocations']:
            sector = alloc['sector']
            sector_weights[sector] = sector_weights.get(sector, 0) + alloc['weight']
        
        # Score de diversificação (baseado no índice Herfindahl)
        weights = [alloc['weight'] for alloc in allocation['allocations']]
        hhi = sum(w**2 for w in weights)
        diversification_score = 1 - hhi
        
        # Análise de concentração
        max_position = max(weights) if weights else 0
        top_3_concentration = sum(sorted(weights, reverse=True)[:3])
        
        return {
            'sector_distribution': sector_weights,
            'diversification_score': float(diversification_score),
            'max_single_position': float(max_position),
            'top_3_concentration': float(top_3_concentration),
            'number_of_sectors': len(sector_weights),
            'diversification_quality': self._classify_diversification(diversification_score)
        }
    
    def _classify_diversification(self, diversification_score: float) -> str:
        """
        Classifica qualidade da diversificação.
        
        Args:
            diversification_score (float): Score de diversificação
            
        Returns:
            str: Classificação
        """
        if diversification_score > 0.8:
            return 'Excelente'
        elif diversification_score > 0.6:
            return 'Boa'
        elif diversification_score > 0.4:
            return 'Moderada'
        else:
            return 'Baixa'
    
    def _empty_recommendations(self) -> Dict[str, Any]:
        """
        Retorna recomendações vazias em caso de erro.
        
        Returns:
            Dict[str, Any]: Recomendações vazias
        """
        return {
            'recommendations': [],
            'suggested_allocation': {'allocations': [], 'total_allocated': 0.0},
            'diversification_analysis': {},
            'error': 'Unable to generate recommendations',
            'generation_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def get_portfolio_optimization(self, current_portfolio: Dict[str, float],
                                 target_return: float = 0.12) -> Dict[str, Any]:
        """
        Otimiza portfólio existente usando teoria moderna de portfólio.
        
        Args:
            current_portfolio (Dict[str, float]): Portfólio atual
            target_return (float): Retorno alvo anual
            
        Returns:
            Dict[str, Any]: Portfólio otimizado
        """
        try:
            # Obtém dados históricos
            market_data = self._get_market_data('2y')
            
            # Filtra apenas ativos do portfólio
            portfolio_tickers = list(current_portfolio.keys())
            available_data = market_data[portfolio_tickers].dropna()
            
            if len(available_data) < 100:  # Mínimo de dados
                return {'error': 'Insufficient data for optimization'}
            
            # Calcula retornos
            returns = available_data.pct_change().dropna()
            
            # Matriz de covariância
            cov_matrix = returns.cov() * 252  # Anualizada
            
            # Retornos esperados (média histórica)
            expected_returns = returns.mean() * 252
            
            # Otimização simples (equal risk contribution)
            n_assets = len(portfolio_tickers)
            equal_weights = np.array([1/n_assets] * n_assets)
            
            # Calcula métricas do portfólio atual
            current_weights = np.array([current_portfolio[ticker] for ticker in portfolio_tickers])
            current_weights = current_weights / current_weights.sum()  # Normaliza
            
            current_return = np.dot(current_weights, expected_returns)
            current_volatility = np.sqrt(np.dot(current_weights.T, np.dot(cov_matrix, current_weights)))
            current_sharpe = (current_return - self.risk_analyzer.risk_free_rate) / current_volatility
            
            # Portfólio otimizado (equal risk)
            optimized_return = np.dot(equal_weights, expected_returns)
            optimized_volatility = np.sqrt(np.dot(equal_weights.T, np.dot(cov_matrix, equal_weights)))
            optimized_sharpe = (optimized_return - self.risk_analyzer.risk_free_rate) / optimized_volatility
            
            # Gera recomendações de rebalanceamento
            rebalancing_suggestions = []
            for i, ticker in enumerate(portfolio_tickers):
                current_weight = current_weights[i]
                suggested_weight = equal_weights[i]
                difference = suggested_weight - current_weight
                
                if abs(difference) > 0.05:  # Só sugere se diferença > 5%
                    action = 'increase' if difference > 0 else 'decrease'
                    rebalancing_suggestions.append({
                        'ticker': ticker,
                        'current_weight': float(current_weight),
                        'suggested_weight': float(suggested_weight),
                        'action': action,
                        'difference': float(difference)
                    })
            
            return {
                'current_portfolio': {
                    'expected_return': float(current_return),
                    'volatility': float(current_volatility),
                    'sharpe_ratio': float(current_sharpe)
                },
                'optimized_portfolio': {
                    'expected_return': float(optimized_return),
                    'volatility': float(optimized_volatility),
                    'sharpe_ratio': float(optimized_sharpe)
                },
                'rebalancing_suggestions': rebalancing_suggestions,
                'improvement': {
                    'return_improvement': float(optimized_return - current_return),
                    'sharpe_improvement': float(optimized_sharpe - current_sharpe)
                },
                'optimization_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Erro na otimização de portfólio: {e}")
            return {'error': f'Portfolio optimization failed: {str(e)}'}
    
    def get_market_regime_analysis(self) -> Dict[str, Any]:
        """
        Analisa regime atual do mercado.
        
        Returns:
            Dict[str, Any]: Análise do regime de mercado
        """
        try:
            # Obtém dados do IBOVESPA
            ibov_data = yf.download('^BVSP', period='1y')['Adj Close']
            
            if len(ibov_data) < 50:
                return {'error': 'Insufficient market data'}
            
            # Calcula retornos
            returns = ibov_data.pct_change().dropna()
            
            # Métricas de regime
            current_volatility = returns.rolling(30).std().iloc[-1] * np.sqrt(252)
            avg_volatility = returns.std() * np.sqrt(252)
            
            # Momentum
            momentum_1m = (ibov_data.iloc[-1] / ibov_data.iloc[-21]) - 1
            momentum_3m = (ibov_data.iloc[-1] / ibov_data.iloc[-63]) - 1
            
            # Tendência
            sma_50 = ibov_data.rolling(50).mean().iloc[-1]
            sma_200 = ibov_data.rolling(200).mean().iloc[-1]
            current_price = ibov_data.iloc[-1]
            
            # Classifica regime
            if current_volatility > avg_volatility * 1.5:
                volatility_regime = 'high'
            elif current_volatility < avg_volatility * 0.7:
                volatility_regime = 'low'
            else:
                volatility_regime = 'normal'
            
            if current_price > sma_50 > sma_200:
                trend_regime = 'bullish'
            elif current_price < sma_50 < sma_200:
                trend_regime = 'bearish'
            else:
                trend_regime = 'sideways'
            
            # Recomendações baseadas no regime
            regime_recommendations = self._get_regime_recommendations(
                volatility_regime, trend_regime, momentum_1m
            )
            
            return {
                'volatility_regime': volatility_regime,
                'trend_regime': trend_regime,
                'current_volatility': float(current_volatility),
                'average_volatility': float(avg_volatility),
                'momentum_1m': float(momentum_1m),
                'momentum_3m': float(momentum_3m),
                'price_vs_sma50': float((current_price / sma_50) - 1),
                'price_vs_sma200': float((current_price / sma_200) - 1),
                'regime_recommendations': regime_recommendations,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
        except Exception as e:
            print(f"Erro na análise de regime: {e}")
            return {'error': f'Market regime analysis failed: {str(e)}'}
    
    def _get_regime_recommendations(self, volatility_regime: str, 
                                  trend_regime: str, momentum: float) -> List[str]:
        """
        Gera recomendações baseadas no regime de mercado.
        
        Args:
            volatility_regime (str): Regime de volatilidade
            trend_regime (str): Regime de tendência
            momentum (float): Momentum de 1 mês
            
        Returns:
            List[str]: Lista de recomendações
        """
        recommendations = []
        
        # Recomendações de volatilidade
        if volatility_regime == 'high':
            recommendations.append("Alta volatilidade: Reduza exposição a ativos de risco")
            recommendations.append("Considere aumentar posição em ativos defensivos")
        elif volatility_regime == 'low':
            recommendations.append("Baixa volatilidade: Oportunidade para aumentar exposição")
            recommendations.append("Ambiente favorável para estratégias de crescimento")
        
        # Recomendações de tendência
        if trend_regime == 'bullish':
            recommendations.append("Tendência de alta: Mantenha exposição a ações")
            recommendations.append("Considere reduzir posição em renda fixa")
        elif trend_regime == 'bearish':
            recommendations.append("Tendência de baixa: Aumente posição defensiva")
            recommendations.append("Considere proteção via hedge ou cash")
        else:
            recommendations.append("Mercado lateral: Estratégias de range-bound")
            recommendations.append("Foque em ativos com dividendos")
        
        # Recomendações de momentum
        if momentum > 0.05:
            recommendations.append("Momentum positivo: Aproveite continuação da alta")
        elif momentum < -0.05:
            recommendations.append("Momentum negativo: Cautela com novas posições")
        
        return recommendations

