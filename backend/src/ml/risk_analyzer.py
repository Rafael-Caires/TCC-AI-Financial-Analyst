"""
Sistema de análise de risco quantitativo para portfólios financeiros

Este módulo implementa métricas avançadas de risco como VaR, CVaR, Sharpe Ratio,
Maximum Drawdown, Beta e Alpha para análise de portfólios.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Any, Optional, Tuple
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class RiskAnalyzer:
    """
    Classe para análise quantitativa de risco de portfólios e ativos individuais.
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        """
        Inicializa o analisador de risco.
        
        Args:
            risk_free_rate (float): Taxa livre de risco anual (padrão: 2%)
        """
        self.risk_free_rate = risk_free_rate
        self.confidence_levels = [0.90, 0.95, 0.99]
        
    def calculate_returns(self, prices: pd.Series) -> pd.Series:
        """
        Calcula retornos logarítmicos de uma série de preços.
        
        Args:
            prices (pd.Series): Série de preços
            
        Returns:
            pd.Series: Série de retornos
        """
        return np.log(prices / prices.shift(1)).dropna()
    
    def calculate_var(self, returns: pd.Series, confidence_level: float = 0.95, 
                     method: str = 'historical') -> float:
        """
        Calcula Value at Risk (VaR).
        
        Args:
            returns (pd.Series): Série de retornos
            confidence_level (float): Nível de confiança
            method (str): Método ('historical', 'parametric', 'monte_carlo')
            
        Returns:
            float: VaR no nível de confiança especificado
        """
        if len(returns) == 0:
            return 0.0
            
        if method == 'historical':
            return np.percentile(returns, (1 - confidence_level) * 100)
        
        elif method == 'parametric':
            mean = returns.mean()
            std = returns.std()
            z_score = stats.norm.ppf(1 - confidence_level)
            return mean + z_score * std
        
        elif method == 'monte_carlo':
            # Simulação Monte Carlo
            mean = returns.mean()
            std = returns.std()
            simulations = np.random.normal(mean, std, 10000)
            return np.percentile(simulations, (1 - confidence_level) * 100)
        
        else:
            raise ValueError(f"Método não suportado: {method}")
    
    def calculate_cvar(self, returns: pd.Series, confidence_level: float = 0.95) -> float:
        """
        Calcula Conditional Value at Risk (CVaR) ou Expected Shortfall.
        
        Args:
            returns (pd.Series): Série de retornos
            confidence_level (float): Nível de confiança
            
        Returns:
            float: CVaR no nível de confiança especificado
        """
        if len(returns) == 0:
            return 0.0
            
        var = self.calculate_var(returns, confidence_level, 'historical')
        return returns[returns <= var].mean()
    
    def calculate_sharpe_ratio(self, returns: pd.Series, 
                             risk_free_rate: Optional[float] = None) -> float:
        """
        Calcula o Sharpe Ratio.
        
        Args:
            returns (pd.Series): Série de retornos
            risk_free_rate (float): Taxa livre de risco (opcional)
            
        Returns:
            float: Sharpe Ratio
        """
        if len(returns) == 0:
            return 0.0
            
        rf_rate = risk_free_rate or self.risk_free_rate
        
        # Converte taxa anual para frequência dos dados
        if len(returns) > 252:  # Dados diários
            rf_daily = rf_rate / 252
        elif len(returns) > 52:  # Dados semanais
            rf_daily = rf_rate / 52
        else:  # Dados mensais
            rf_daily = rf_rate / 12
        
        excess_returns = returns - rf_daily
        
        if excess_returns.std() == 0:
            return 0.0
            
        return excess_returns.mean() / excess_returns.std()
    
    def calculate_sortino_ratio(self, returns: pd.Series, 
                              risk_free_rate: Optional[float] = None) -> float:
        """
        Calcula o Sortino Ratio (considera apenas volatilidade negativa).
        
        Args:
            returns (pd.Series): Série de retornos
            risk_free_rate (float): Taxa livre de risco (opcional)
            
        Returns:
            float: Sortino Ratio
        """
        if len(returns) == 0:
            return 0.0
            
        rf_rate = risk_free_rate or self.risk_free_rate
        
        # Converte taxa anual para frequência dos dados
        if len(returns) > 252:
            rf_daily = rf_rate / 252
        elif len(returns) > 52:
            rf_daily = rf_rate / 52
        else:
            rf_daily = rf_rate / 12
        
        excess_returns = returns - rf_daily
        negative_returns = excess_returns[excess_returns < 0]
        
        if len(negative_returns) == 0 or negative_returns.std() == 0:
            return 0.0
            
        return excess_returns.mean() / negative_returns.std()
    
    def calculate_maximum_drawdown(self, prices: pd.Series) -> Dict[str, Any]:
        """
        Calcula Maximum Drawdown e informações relacionadas.
        
        Args:
            prices (pd.Series): Série de preços
            
        Returns:
            Dict[str, Any]: Informações sobre drawdown
        """
        if len(prices) == 0:
            return {
                'max_drawdown': 0.0,
                'max_drawdown_duration': 0,
                'current_drawdown': 0.0,
                'peak_date': None,
                'trough_date': None
            }
        
        # Calcula picos cumulativos
        cumulative_max = prices.expanding().max()
        
        # Calcula drawdowns
        drawdowns = (prices - cumulative_max) / cumulative_max
        
        # Maximum drawdown
        max_drawdown = drawdowns.min()
        
        # Encontra datas do pico e vale
        max_dd_date = drawdowns.idxmin()
        peak_date = cumulative_max[:max_dd_date].idxmax()
        
        # Calcula duração do maximum drawdown
        recovery_mask = prices[max_dd_date:] >= prices[peak_date]
        if recovery_mask.any():
            recovery_date = prices[max_dd_date:][recovery_mask].index[0]
            max_dd_duration = (recovery_date - peak_date).days
        else:
            max_dd_duration = (prices.index[-1] - peak_date).days
        
        # Drawdown atual
        current_drawdown = drawdowns.iloc[-1]
        
        return {
            'max_drawdown': float(max_drawdown),
            'max_drawdown_duration': max_dd_duration,
            'current_drawdown': float(current_drawdown),
            'peak_date': peak_date.strftime('%Y-%m-%d') if peak_date else None,
            'trough_date': max_dd_date.strftime('%Y-%m-%d') if max_dd_date else None
        }
    
    def calculate_beta_alpha(self, asset_returns: pd.Series, 
                           market_returns: pd.Series) -> Dict[str, float]:
        """
        Calcula Beta e Alpha em relação ao mercado.
        
        Args:
            asset_returns (pd.Series): Retornos do ativo
            market_returns (pd.Series): Retornos do mercado (benchmark)
            
        Returns:
            Dict[str, float]: Beta e Alpha
        """
        if len(asset_returns) == 0 or len(market_returns) == 0:
            return {'beta': 0.0, 'alpha': 0.0, 'r_squared': 0.0}
        
        # Alinha as séries temporais
        aligned_data = pd.concat([asset_returns, market_returns], axis=1, join='inner')
        aligned_data.columns = ['asset', 'market']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) < 2:
            return {'beta': 0.0, 'alpha': 0.0, 'r_squared': 0.0}
        
        # Regressão linear
        slope, intercept, r_value, p_value, std_err = stats.linregress(
            aligned_data['market'], aligned_data['asset']
        )
        
        beta = slope
        alpha = intercept
        r_squared = r_value ** 2
        
        return {
            'beta': float(beta),
            'alpha': float(alpha),
            'r_squared': float(r_squared)
        }
    
    def calculate_tracking_error(self, asset_returns: pd.Series, 
                               benchmark_returns: pd.Series) -> float:
        """
        Calcula Tracking Error em relação a um benchmark.
        
        Args:
            asset_returns (pd.Series): Retornos do ativo
            benchmark_returns (pd.Series): Retornos do benchmark
            
        Returns:
            float: Tracking Error
        """
        # Alinha as séries
        aligned_data = pd.concat([asset_returns, benchmark_returns], axis=1, join='inner')
        aligned_data.columns = ['asset', 'benchmark']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) == 0:
            return 0.0
        
        # Calcula diferença de retornos
        excess_returns = aligned_data['asset'] - aligned_data['benchmark']
        
        return float(excess_returns.std())
    
    def calculate_information_ratio(self, asset_returns: pd.Series, 
                                  benchmark_returns: pd.Series) -> float:
        """
        Calcula Information Ratio.
        
        Args:
            asset_returns (pd.Series): Retornos do ativo
            benchmark_returns (pd.Series): Retornos do benchmark
            
        Returns:
            float: Information Ratio
        """
        # Alinha as séries
        aligned_data = pd.concat([asset_returns, benchmark_returns], axis=1, join='inner')
        aligned_data.columns = ['asset', 'benchmark']
        aligned_data = aligned_data.dropna()
        
        if len(aligned_data) == 0:
            return 0.0
        
        # Calcula diferença de retornos
        excess_returns = aligned_data['asset'] - aligned_data['benchmark']
        
        if excess_returns.std() == 0:
            return 0.0
        
        return float(excess_returns.mean() / excess_returns.std())
    
    def analyze_portfolio_risk(self, portfolio_data: Dict[str, float], 
                             period_days: int = 252) -> Dict[str, Any]:
        """
        Análise completa de risco de um portfólio.
        
        Args:
            portfolio_data (Dict[str, float]): {ticker: peso} do portfólio
            period_days (int): Período de análise em dias
            
        Returns:
            Dict[str, Any]: Análise completa de risco
        """
        try:
            # Baixa dados históricos
            end_date = datetime.now()
            start_date = end_date - timedelta(days=period_days + 50)  # Buffer para cálculos
            
            tickers = list(portfolio_data.keys())
            weights = np.array(list(portfolio_data.values()))
            
            # Normaliza pesos
            weights = weights / weights.sum()
            
            # Baixa dados
            data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
            
            if isinstance(data, pd.Series):
                data = data.to_frame(tickers[0])
            
            # Remove dados faltantes
            data = data.dropna()
            
            if len(data) < 30:  # Mínimo de dados necessários
                return self._empty_risk_analysis()
            
            # Calcula retornos
            returns = data.pct_change().dropna()
            
            # Retornos do portfólio
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Baixa dados do mercado (IBOV como proxy)
            market_data = yf.download('^BVSP', start=start_date, end=end_date)['Adj Close']
            market_returns = market_data.pct_change().dropna()
            
            # Análise de risco individual
            individual_analysis = {}
            for i, ticker in enumerate(tickers):
                if ticker in returns.columns:
                    asset_returns = returns[ticker].dropna()
                    
                    individual_analysis[ticker] = {
                        'weight': float(weights[i]),
                        'var_95': float(self.calculate_var(asset_returns, 0.95)),
                        'cvar_95': float(self.calculate_cvar(asset_returns, 0.95)),
                        'sharpe_ratio': float(self.calculate_sharpe_ratio(asset_returns)),
                        'sortino_ratio': float(self.calculate_sortino_ratio(asset_returns)),
                        'volatility': float(asset_returns.std()),
                        'max_drawdown': self.calculate_maximum_drawdown(data[ticker])
                    }
                    
                    # Beta e Alpha em relação ao mercado
                    beta_alpha = self.calculate_beta_alpha(asset_returns, market_returns)
                    individual_analysis[ticker].update(beta_alpha)
            
            # Análise do portfólio
            portfolio_analysis = {
                'var_90': float(self.calculate_var(portfolio_returns, 0.90)),
                'var_95': float(self.calculate_var(portfolio_returns, 0.95)),
                'var_99': float(self.calculate_var(portfolio_returns, 0.99)),
                'cvar_95': float(self.calculate_cvar(portfolio_returns, 0.95)),
                'sharpe_ratio': float(self.calculate_sharpe_ratio(portfolio_returns)),
                'sortino_ratio': float(self.calculate_sortino_ratio(portfolio_returns)),
                'volatility': float(portfolio_returns.std()),
                'annualized_volatility': float(portfolio_returns.std() * np.sqrt(252)),
                'expected_return': float(portfolio_returns.mean() * 252),
                'tracking_error': float(self.calculate_tracking_error(portfolio_returns, market_returns)),
                'information_ratio': float(self.calculate_information_ratio(portfolio_returns, market_returns))
            }
            
            # Maximum drawdown do portfólio
            portfolio_prices = (1 + portfolio_returns).cumprod()
            portfolio_analysis['max_drawdown'] = self.calculate_maximum_drawdown(portfolio_prices)
            
            # Beta e Alpha do portfólio
            portfolio_beta_alpha = self.calculate_beta_alpha(portfolio_returns, market_returns)
            portfolio_analysis.update(portfolio_beta_alpha)
            
            # Matriz de correlação
            correlation_matrix = returns[tickers].corr().to_dict()
            
            # Análise de concentração
            concentration_analysis = self._analyze_concentration(weights, tickers)
            
            # Stress testing
            stress_test = self._perform_stress_test(portfolio_returns)
            
            return {
                'portfolio_analysis': portfolio_analysis,
                'individual_analysis': individual_analysis,
                'correlation_matrix': correlation_matrix,
                'concentration_analysis': concentration_analysis,
                'stress_test': stress_test,
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'period_analyzed': f'{len(portfolio_returns)} days'
            }
            
        except Exception as e:
            print(f"Erro na análise de risco: {e}")
            return self._empty_risk_analysis()
    
    def _empty_risk_analysis(self) -> Dict[str, Any]:
        """
        Retorna análise vazia em caso de erro.
        
        Returns:
            Dict[str, Any]: Análise vazia
        """
        return {
            'portfolio_analysis': {},
            'individual_analysis': {},
            'correlation_matrix': {},
            'concentration_analysis': {},
            'stress_test': {},
            'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'error': 'Insufficient data for analysis'
        }
    
    def _analyze_concentration(self, weights: np.ndarray, 
                             tickers: List[str]) -> Dict[str, Any]:
        """
        Analisa concentração do portfólio.
        
        Args:
            weights (np.ndarray): Pesos dos ativos
            tickers (List[str]): Lista de tickers
            
        Returns:
            Dict[str, Any]: Análise de concentração
        """
        # Índice Herfindahl-Hirschman
        hhi = np.sum(weights ** 2)
        
        # Número efetivo de ativos
        effective_assets = 1 / hhi if hhi > 0 else 0
        
        # Concentração dos top 3 ativos
        top_3_concentration = np.sum(np.sort(weights)[-3:])
        
        # Maior peso individual
        max_weight = np.max(weights)
        max_weight_ticker = tickers[np.argmax(weights)]
        
        return {
            'herfindahl_index': float(hhi),
            'effective_number_assets': float(effective_assets),
            'top_3_concentration': float(top_3_concentration),
            'max_individual_weight': float(max_weight),
            'max_weight_ticker': max_weight_ticker,
            'diversification_ratio': float(1 - hhi)
        }
    
    def _perform_stress_test(self, returns: pd.Series) -> Dict[str, Any]:
        """
        Realiza stress test do portfólio.
        
        Args:
            returns (pd.Series): Retornos do portfólio
            
        Returns:
            Dict[str, Any]: Resultados do stress test
        """
        if len(returns) == 0:
            return {}
        
        # Cenários de stress
        scenarios = {
            'market_crash_2008': -0.20,  # Queda de 20% em um dia
            'covid_crash_2020': -0.12,   # Queda de 12% em um dia
            'flash_crash': -0.10,        # Queda de 10% em um dia
            'moderate_correction': -0.05  # Queda de 5% em um dia
        }
        
        current_volatility = returns.std()
        
        stress_results = {}
        
        for scenario_name, shock in scenarios.items():
            # Calcula probabilidade baseada na distribuição histórica
            probability = len(returns[returns <= shock]) / len(returns)
            
            # Simula impacto no portfólio
            stressed_return = shock
            
            stress_results[scenario_name] = {
                'shock_magnitude': float(shock),
                'historical_probability': float(probability),
                'expected_loss': float(stressed_return),
                'probability_category': self._categorize_probability(probability)
            }
        
        # Teste de volatilidade extrema
        extreme_vol_scenarios = {
            'double_volatility': current_volatility * 2,
            'triple_volatility': current_volatility * 3
        }
        
        for scenario_name, vol in extreme_vol_scenarios.items():
            var_extreme = returns.mean() - 2.33 * vol  # VaR 99%
            stress_results[scenario_name] = {
                'volatility_multiplier': float(vol / current_volatility),
                'var_99_extreme': float(var_extreme),
                'scenario_type': 'volatility_stress'
            }
        
        return stress_results
    
    def _categorize_probability(self, probability: float) -> str:
        """
        Categoriza probabilidade de eventos.
        
        Args:
            probability (float): Probabilidade do evento
            
        Returns:
            str: Categoria da probabilidade
        """
        if probability >= 0.1:
            return 'high'
        elif probability >= 0.05:
            return 'medium'
        elif probability >= 0.01:
            return 'low'
        else:
            return 'very_low'
    
    def generate_risk_report(self, portfolio_data: Dict[str, float]) -> str:
        """
        Gera relatório textual de risco.
        
        Args:
            portfolio_data (Dict[str, float]): Dados do portfólio
            
        Returns:
            str: Relatório de risco
        """
        analysis = self.analyze_portfolio_risk(portfolio_data)
        
        if 'error' in analysis:
            return "Erro: Dados insuficientes para análise de risco."
        
        portfolio = analysis['portfolio_analysis']
        
        report = f"""
RELATÓRIO DE ANÁLISE DE RISCO
=============================

Data da Análise: {analysis['analysis_date']}
Período Analisado: {analysis['period_analyzed']}

MÉTRICAS PRINCIPAIS DO PORTFÓLIO:
- VaR 95%: {portfolio.get('var_95', 0):.2%}
- CVaR 95%: {portfolio.get('cvar_95', 0):.2%}
- Sharpe Ratio: {portfolio.get('sharpe_ratio', 0):.3f}
- Volatilidade Anualizada: {portfolio.get('annualized_volatility', 0):.2%}
- Maximum Drawdown: {portfolio.get('max_drawdown', {}).get('max_drawdown', 0):.2%}
- Beta: {portfolio.get('beta', 0):.3f}

ANÁLISE DE CONCENTRAÇÃO:
- Índice Herfindahl: {analysis['concentration_analysis'].get('herfindahl_index', 0):.3f}
- Número Efetivo de Ativos: {analysis['concentration_analysis'].get('effective_number_assets', 0):.1f}
- Concentração Top 3: {analysis['concentration_analysis'].get('top_3_concentration', 0):.2%}

CLASSIFICAÇÃO DE RISCO:
{self._classify_portfolio_risk(portfolio)}

RECOMENDAÇÕES:
{self._generate_risk_recommendations(analysis)}
        """
        
        return report.strip()
    
    def _classify_portfolio_risk(self, portfolio_analysis: Dict[str, Any]) -> str:
        """
        Classifica o nível de risco do portfólio.
        
        Args:
            portfolio_analysis (Dict[str, Any]): Análise do portfólio
            
        Returns:
            str: Classificação de risco
        """
        var_95 = abs(portfolio_analysis.get('var_95', 0))
        volatility = portfolio_analysis.get('annualized_volatility', 0)
        max_dd = abs(portfolio_analysis.get('max_drawdown', {}).get('max_drawdown', 0))
        
        # Pontuação de risco (0-100)
        risk_score = 0
        
        # VaR component (0-30)
        if var_95 > 0.05:
            risk_score += 30
        elif var_95 > 0.03:
            risk_score += 20
        elif var_95 > 0.02:
            risk_score += 10
        
        # Volatility component (0-40)
        if volatility > 0.30:
            risk_score += 40
        elif volatility > 0.20:
            risk_score += 30
        elif volatility > 0.15:
            risk_score += 20
        elif volatility > 0.10:
            risk_score += 10
        
        # Drawdown component (0-30)
        if max_dd > 0.30:
            risk_score += 30
        elif max_dd > 0.20:
            risk_score += 20
        elif max_dd > 0.10:
            risk_score += 10
        
        # Classificação
        if risk_score >= 70:
            return "ALTO RISCO - Portfólio com alta volatilidade e potencial para perdas significativas"
        elif risk_score >= 40:
            return "RISCO MODERADO - Portfólio balanceado com risco controlado"
        else:
            return "BAIXO RISCO - Portfólio conservador com baixa volatilidade"
    
    def _generate_risk_recommendations(self, analysis: Dict[str, Any]) -> str:
        """
        Gera recomendações baseadas na análise de risco.
        
        Args:
            analysis (Dict[str, Any]): Análise completa
            
        Returns:
            str: Recomendações
        """
        recommendations = []
        
        portfolio = analysis['portfolio_analysis']
        concentration = analysis['concentration_analysis']
        
        # Recomendações de diversificação
        if concentration.get('herfindahl_index', 0) > 0.25:
            recommendations.append("- Considere diversificar mais o portfólio para reduzir concentração")
        
        if concentration.get('max_individual_weight', 0) > 0.30:
            recommendations.append("- Reduza a exposição ao ativo com maior peso individual")
        
        # Recomendações de risco
        if abs(portfolio.get('var_95', 0)) > 0.05:
            recommendations.append("- VaR elevado: considere reduzir exposição a ativos de alto risco")
        
        if portfolio.get('sharpe_ratio', 0) < 0.5:
            recommendations.append("- Sharpe Ratio baixo: busque ativos com melhor relação risco-retorno")
        
        if abs(portfolio.get('max_drawdown', {}).get('max_drawdown', 0)) > 0.20:
            recommendations.append("- Maximum Drawdown alto: implemente estratégias de stop-loss")
        
        if not recommendations:
            recommendations.append("- Portfólio apresenta perfil de risco adequado")
        
        return "\n".join(recommendations)

