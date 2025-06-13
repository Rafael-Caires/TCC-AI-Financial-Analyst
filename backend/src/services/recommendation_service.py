"""
Serviço de recomendações de investimento

Este arquivo implementa serviços para gerar recomendações de investimento
baseadas em previsões de modelos de ML e perfil do usuário.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

class RecommendationService:
    """
    Serviço para gerar recomendações de investimento.
    """
    
    @staticmethod
    def generate_stock_recommendation(prediction_data, user_risk_profile, current_price=None):
        """
        Gera uma recomendação para um ativo específico com base nas previsões.
        
        Args:
            prediction_data (dict): Dados de previsão do modelo
            user_risk_profile (str): Perfil de risco do usuário ('conservador', 'moderado', 'arrojado')
            current_price (float): Preço atual do ativo (opcional)
            
        Returns:
            dict: Recomendação de investimento
        """
        # Verifica se há dados de previsão
        if not prediction_data or 'predictions' not in prediction_data or not prediction_data['predictions']:
            raise ValueError("Dados de previsão inválidos")
        
        # Obtém o preço atual
        if current_price is None:
            current_price = prediction_data['last_price']
        
        # Extrai previsões
        predictions = prediction_data['predictions']
        
        # Calcula a tendência de curto prazo (7 dias)
        short_term_prices = [p['predicted_price'] for p in predictions[:7]]
        short_term_trend = np.mean(short_term_prices) / current_price - 1
        
        # Calcula a tendência de médio prazo (30 dias ou o máximo disponível)
        medium_term_prices = [p['predicted_price'] for p in predictions[:min(30, len(predictions))]]
        medium_term_trend = np.mean(medium_term_prices) / current_price - 1
        
        # Calcula a volatilidade prevista
        predicted_volatility = np.std([p['predicted_price'] for p in predictions[:min(30, len(predictions))]])
        volatility_ratio = predicted_volatility / current_price
        
        # Calcula o potencial de alta e baixa
        upside_potential = max([p['upper_bound'] for p in predictions[:min(30, len(predictions))]])
        downside_risk = min([p['lower_bound'] for p in predictions[:min(30, len(predictions))]])
        
        upside_ratio = upside_potential / current_price - 1
        downside_ratio = 1 - downside_risk / current_price
        
        # Calcula o risco-retorno
        risk_reward_ratio = upside_ratio / downside_ratio if downside_ratio > 0 else 0
        
        # Determina a ação recomendada com base no perfil de risco
        action, confidence, reason = RecommendationService._determine_action(
            short_term_trend, 
            medium_term_trend, 
            volatility_ratio, 
            risk_reward_ratio, 
            user_risk_profile
        )
        
        # Formata a recomendação
        recommendation = {
            'ticker': prediction_data['ticker'],
            'current_price': current_price,
            'action': action,
            'confidence': confidence,
            'reason': reason,
            'analysis': {
                'short_term_trend': short_term_trend * 100,  # Converte para percentual
                'medium_term_trend': medium_term_trend * 100,
                'volatility_ratio': volatility_ratio * 100,
                'upside_potential': upside_ratio * 100,
                'downside_risk': downside_ratio * 100,
                'risk_reward_ratio': risk_reward_ratio
            },
            'prediction_summary': {
                'short_term': np.mean(short_term_prices),
                'medium_term': np.mean(medium_term_prices),
                'max_price': max([p['predicted_price'] for p in predictions]),
                'min_price': min([p['predicted_price'] for p in predictions])
            },
            'timestamp': datetime.now().isoformat()
        }
        
        return recommendation
    
    @staticmethod
    def _determine_action(short_term_trend, medium_term_trend, volatility_ratio, risk_reward_ratio, risk_profile):
        """
        Determina a ação recomendada com base nos indicadores e perfil de risco.
        
        Args:
            short_term_trend (float): Tendência de curto prazo
            medium_term_trend (float): Tendência de médio prazo
            volatility_ratio (float): Razão de volatilidade
            risk_reward_ratio (float): Razão risco-retorno
            risk_profile (str): Perfil de risco do usuário
            
        Returns:
            tuple: (ação, confiança, razão)
        """
        # Ajusta os limiares com base no perfil de risco
        if risk_profile == 'conservador':
            trend_threshold = 0.05  # 5%
            volatility_threshold = 0.03  # 3%
            risk_reward_threshold = 2.0
        elif risk_profile == 'moderado':
            trend_threshold = 0.03  # 3%
            volatility_threshold = 0.05  # 5%
            risk_reward_threshold = 1.5
        else:  # arrojado
            trend_threshold = 0.02  # 2%
            volatility_threshold = 0.08  # 8%
            risk_reward_threshold = 1.2
        
        # Lógica de decisão
        if medium_term_trend > trend_threshold and short_term_trend > 0:
            # Tendência de alta forte
            if risk_reward_ratio > risk_reward_threshold:
                action = 'comprar'
                confidence = min(0.9, 0.5 + medium_term_trend + risk_reward_ratio / 10)
                reason = "Forte tendência de alta com boa relação risco-retorno"
            else:
                action = 'manter'
                confidence = 0.6
                reason = "Tendência de alta, mas relação risco-retorno não ideal"
        
        elif medium_term_trend > 0 and short_term_trend > 0:
            # Tendência de alta moderada
            if risk_reward_ratio > risk_reward_threshold:
                action = 'comprar'
                confidence = min(0.8, 0.4 + medium_term_trend + risk_reward_ratio / 10)
                reason = "Tendência de alta moderada com boa relação risco-retorno"
            else:
                action = 'manter'
                confidence = 0.5
                reason = "Tendência de alta moderada, monitorar para oportunidade de compra"
        
        elif medium_term_trend < -trend_threshold and short_term_trend < 0:
            # Tendência de baixa forte
            action = 'vender'
            confidence = min(0.9, 0.5 + abs(medium_term_trend))
            reason = "Forte tendência de baixa, potencial de queda significativo"
        
        elif medium_term_trend < 0 and short_term_trend < 0:
            # Tendência de baixa moderada
            if volatility_ratio > volatility_threshold:
                action = 'vender'
                confidence = min(0.8, 0.4 + abs(medium_term_trend))
                reason = "Tendência de baixa com alta volatilidade"
            else:
                action = 'manter'
                confidence = 0.5
                reason = "Tendência de baixa moderada, monitorar para possível saída"
        
        else:
            # Tendência lateral ou mista
            if volatility_ratio > volatility_threshold * 1.5:
                action = 'manter'
                confidence = 0.6
                reason = "Mercado lateral com alta volatilidade, aguardar definição de tendência"
            else:
                action = 'manter'
                confidence = 0.7
                reason = "Mercado lateral, sem tendência clara no momento"
        
        return action, confidence, reason
    
    @staticmethod
    def generate_portfolio_recommendations(portfolio_assets, predictions, user_risk_profile):
        """
        Gera recomendações para um portfólio de ativos.
        
        Args:
            portfolio_assets (list): Lista de ativos no portfólio
            predictions (dict): Dicionário com previsões para cada ativo
            user_risk_profile (str): Perfil de risco do usuário
            
        Returns:
            dict: Recomendações para o portfólio
        """
        recommendations = []
        
        # Gera recomendações para cada ativo
        for asset in portfolio_assets:
            ticker = asset['ticker']
            
            if ticker in predictions:
                try:
                    recommendation = RecommendationService.generate_stock_recommendation(
                        predictions[ticker],
                        user_risk_profile,
                        asset.get('current_price')
                    )
                    
                    # Adiciona informações do portfólio
                    recommendation['portfolio_info'] = {
                        'quantity': asset.get('quantity', 0),
                        'purchase_price': asset.get('purchase_price', 0),
                        'current_value': asset.get('quantity', 0) * recommendation['current_price'],
                        'profit_loss': (recommendation['current_price'] / asset.get('purchase_price', 1) - 1) * 100
                    }
                    
                    recommendations.append(recommendation)
                except Exception as e:
                    print(f"Erro ao gerar recomendação para {ticker}: {str(e)}")
        
        # Calcula a alocação recomendada
        allocation = RecommendationService._calculate_recommended_allocation(
            recommendations, 
            user_risk_profile
        )
        
        return {
            'recommendations': recommendations,
            'allocation': allocation,
            'summary': RecommendationService._generate_portfolio_summary(recommendations, allocation),
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def _calculate_recommended_allocation(recommendations, risk_profile):
        """
        Calcula a alocação recomendada para o portfólio.
        
        Args:
            recommendations (list): Lista de recomendações
            risk_profile (str): Perfil de risco do usuário
            
        Returns:
            dict: Alocação recomendada
        """
        # Define a alocação base por perfil de risco
        if risk_profile == 'conservador':
            base_allocation = {
                'acoes': 0.30,
                'renda_fixa': 0.50,
                'tesouro': 0.15,
                'caixa': 0.05
            }
        elif risk_profile == 'moderado':
            base_allocation = {
                'acoes': 0.50,
                'renda_fixa': 0.30,
                'tesouro': 0.10,
                'caixa': 0.10
            }
        else:  # arrojado
            base_allocation = {
                'acoes': 0.70,
                'renda_fixa': 0.15,
                'tesouro': 0.05,
                'caixa': 0.10
            }
        
        # Ajusta a alocação com base nas recomendações
        buy_count = sum(1 for r in recommendations if r['action'] == 'comprar')
        sell_count = sum(1 for r in recommendations if r['action'] == 'vender')
        
        adjusted_allocation = base_allocation.copy()
        
        # Se há mais recomendações de compra, aumenta a alocação em ações
        if buy_count > sell_count and len(recommendations) > 0:
            buy_ratio = buy_count / len(recommendations)
            adjustment = min(0.10, buy_ratio * 0.20)  # Ajuste máximo de 10%
            
            adjusted_allocation['acoes'] = min(0.90, adjusted_allocation['acoes'] + adjustment)
            adjusted_allocation['renda_fixa'] = max(0.05, adjusted_allocation['renda_fixa'] - adjustment * 0.7)
            adjusted_allocation['tesouro'] = max(0.02, adjusted_allocation['tesouro'] - adjustment * 0.3)
        
        # Se há mais recomendações de venda, diminui a alocação em ações
        elif sell_count > buy_count and len(recommendations) > 0:
            sell_ratio = sell_count / len(recommendations)
            adjustment = min(0.15, sell_ratio * 0.25)  # Ajuste máximo de 15%
            
            adjusted_allocation['acoes'] = max(0.10, adjusted_allocation['acoes'] - adjustment)
            adjusted_allocation['renda_fixa'] = min(0.70, adjusted_allocation['renda_fixa'] + adjustment * 0.5)
            adjusted_allocation['tesouro'] = min(0.30, adjusted_allocation['tesouro'] + adjustment * 0.3)
            adjusted_allocation['caixa'] = min(0.20, adjusted_allocation['caixa'] + adjustment * 0.2)
        
        # Normaliza para garantir que a soma seja 1
        total = sum(adjusted_allocation.values())
        for key in adjusted_allocation:
            adjusted_allocation[key] /= total
        
        return adjusted_allocation
    
    @staticmethod
    def _generate_portfolio_summary(recommendations, allocation):
        """
        Gera um resumo das recomendações do portfólio.
        
        Args:
            recommendations (list): Lista de recomendações
            allocation (dict): Alocação recomendada
            
        Returns:
            dict: Resumo do portfólio
        """
        # Conta as ações por tipo
        actions = {
            'comprar': sum(1 for r in recommendations if r['action'] == 'comprar'),
            'manter': sum(1 for r in recommendations if r['action'] == 'manter'),
            'vender': sum(1 for r in recommendations if r['action'] == 'vender')
        }
        
        # Calcula o potencial médio de alta e baixa
        if recommendations:
            avg_upside = np.mean([r['analysis']['upside_potential'] for r in recommendations])
            avg_downside = np.mean([r['analysis']['downside_risk'] for r in recommendations])
            avg_volatility = np.mean([r['analysis']['volatility_ratio'] for r in recommendations])
        else:
            avg_upside = 0
            avg_downside = 0
            avg_volatility = 0
        
        # Gera o resumo
        summary = {
            'actions': actions,
            'potential': {
                'upside': avg_upside,
                'downside': avg_downside,
                'volatility': avg_volatility
            },
            'allocation': allocation,
            'message': RecommendationService._generate_summary_message(actions, allocation)
        }
        
        return summary
    
    @staticmethod
    def _generate_summary_message(actions, allocation):
        """
        Gera uma mensagem de resumo para o portfólio.
        
        Args:
            actions (dict): Contagem de ações por tipo
            allocation (dict): Alocação recomendada
            
        Returns:
            str: Mensagem de resumo
        """
        total_actions = sum(actions.values())
        
        if total_actions == 0:
            return "Não há ativos suficientes para análise de portfólio."
        
        buy_ratio = actions['comprar'] / total_actions if total_actions > 0 else 0
        sell_ratio = actions['vender'] / total_actions if total_actions > 0 else 0
        
        if buy_ratio > 0.6:
            message = "Cenário otimista para o mercado. Recomendamos aumentar a exposição em ações."
        elif sell_ratio > 0.6:
            message = "Cenário de cautela para o mercado. Recomendamos reduzir a exposição em ações e aumentar reserva de segurança."
        elif buy_ratio > sell_ratio:
            message = "Cenário moderadamente positivo. Mantenha a alocação em ações com foco em oportunidades específicas."
        elif sell_ratio > buy_ratio:
            message = "Cenário moderadamente negativo. Considere proteger parte do portfólio com ativos mais seguros."
        else:
            message = "Cenário neutro. Mantenha a diversificação e monitore o mercado para novas oportunidades."
        
        return message
    
    @staticmethod
    def generate_market_recommendations(market_data, user_risk_profile):
        """
        Gera recomendações gerais de mercado.
        
        Args:
            market_data (dict): Dados de mercado
            user_risk_profile (str): Perfil de risco do usuário
            
        Returns:
            dict: Recomendações de mercado
        """
        # Analisa a tendência dos principais índices
        indices_trend = 0
        
        for index_name, index_data in market_data.get('indices', {}).items():
            if index_data.get('change_direction') == 'up':
                indices_trend += 1
            elif index_data.get('change_direction') == 'down':
                indices_trend -= 1
        
        # Analisa o desempenho dos setores
        sectors = market_data.get('sectors', {})
        positive_sectors = [sector for sector, data in sectors.items() if data.get('change_direction') == 'up']
        negative_sectors = [sector for sector, data in sectors.items() if data.get('change_direction') == 'down']
        
        # Determina a postura de mercado
        if indices_trend > 1 and len(positive_sectors) > len(negative_sectors) * 2:
            market_stance = 'bullish'
            confidence = 0.8
        elif indices_trend > 0 and len(positive_sectors) > len(negative_sectors):
            market_stance = 'moderately_bullish'
            confidence = 0.6
        elif indices_trend < -1 and len(negative_sectors) > len(positive_sectors) * 2:
            market_stance = 'bearish'
            confidence = 0.8
        elif indices_trend < 0 and len(negative_sectors) > len(positive_sectors):
            market_stance = 'moderately_bearish'
            confidence = 0.6
        else:
            market_stance = 'neutral'
            confidence = 0.7
        
        # Gera recomendações com base na postura de mercado e perfil de risco
        recommendations = RecommendationService._generate_market_stance_recommendations(
            market_stance, 
            confidence, 
            user_risk_profile,
            positive_sectors,
            negative_sectors
        )
        
        return {
            'market_stance': market_stance,
            'confidence': confidence,
            'recommendations': recommendations,
            'positive_sectors': positive_sectors,
            'negative_sectors': negative_sectors,
            'timestamp': datetime.now().isoformat()
        }
    
    @staticmethod
    def _generate_market_stance_recommendations(market_stance, confidence, risk_profile, positive_sectors, negative_sectors):
        """
        Gera recomendações com base na postura de mercado.
        
        Args:
            market_stance (str): Postura de mercado
            confidence (float): Nível de confiança
            risk_profile (str): Perfil de risco do usuário
            positive_sectors (list): Setores com desempenho positivo
            negative_sectors (list): Setores com desempenho negativo
            
        Returns:
            list: Lista de recomendações
        """
        recommendations = []
        
        # Recomendações gerais com base na postura de mercado
        if market_stance == 'bullish':
            if risk_profile == 'conservador':
                recommendations.append({
                    'type': 'allocation',
                    'action': 'Aumente moderadamente a exposição em ações',
                    'description': 'O mercado está em tendência de alta. Considere aumentar gradualmente sua exposição em ações de qualidade.'
                })
            elif risk_profile == 'moderado':
                recommendations.append({
                    'type': 'allocation',
                    'action': 'Aumente a exposição em ações',
                    'description': 'O mercado está em tendência de alta. Aproveite para aumentar sua exposição em ações com bom potencial de valorização.'
                })
            else:  # arrojado
                recommendations.append({
                    'type': 'allocation',
                    'action': 'Maximize a exposição em ações',
                    'description': 'O mercado está em forte tendência de alta. Considere maximizar sua exposição em ações com alto potencial de crescimento.'
                })
        
        elif market_stance == 'moderately_bullish':
            if risk_profile == 'conservador':
                recommendations.append({
                    'type': 'allocation',
                    'action': 'Mantenha a alocação atual com viés de alta',
                    'description': 'O mercado está moderadamente positivo. Mantenha sua alocação atual, mas considere oportunidades específicas em setores fortes.'
                })
            else:  # moderado ou arrojado
                recommendations.append({
                    'type': 'allocation',
                    'action': 'Aumente moderadamente a exposição em ações',
                    'description': 'O mercado está moderadamente positivo. Considere aumentar sua exposição em ações de setores com bom desempenho.'
                })
        
        elif market_stance == 'bearish':
            if risk_profile == 'conservador':
                recommendations.append({
                    'type': 'allocation',
                    'action': 'Reduza significativamente a exposição em ações',
                    'description': 'O mercado está em tendência de baixa. Considere reduzir sua exposição em ações e aumentar posições em ativos de menor risco.'
                })
            elif risk_profile == 'moderado':
                recommendations.append({
                    'type': 'allocation',
                    'action': 'Reduza a exposição em ações',
                    'description': 'O mercado está em tendência de baixa. Considere reduzir sua exposição em ações e proteger parte do capital.'
                })
            else:  # arrojado
                recommendations.append({
                    'type': 'allocation',
                    'action': 'Reduza moderadamente a exposição em ações',
                    'description': 'O mercado está em tendência de baixa. Considere reduzir parcialmente sua exposição em ações e buscar oportunidades específicas.'
                })
        
        elif market_stance == 'moderately_bearish':
            if risk_profile == 'conservador':
                recommendations.append({
                    'type': 'allocation',
                    'action': 'Reduza a exposição em ações',
                    'description': 'O mercado está moderadamente negativo. Considere reduzir sua exposição em ações e aumentar posições defensivas.'
                })
            else:  # moderado ou arrojado
                recommendations.append({
                    'type': 'allocation',
                    'action': 'Mantenha a alocação atual com viés de cautela',
                    'description': 'O mercado está moderadamente negativo. Mantenha cautela e considere proteger posições em setores mais fracos.'
                })
        
        else:  # neutral
            recommendations.append({
                'type': 'allocation',
                'action': 'Mantenha a diversificação atual',
                'description': 'O mercado está sem tendência clara. Mantenha uma carteira diversificada e monitore oportunidades específicas.'
            })
        
        # Recomendações de setores
        if positive_sectors:
            top_sectors = positive_sectors[:3] if len(positive_sectors) > 3 else positive_sectors
            recommendations.append({
                'type': 'sectors',
                'action': 'Considere exposição nos setores em alta',
                'description': f'Os setores {", ".join(top_sectors)} estão apresentando bom desempenho. Considere aumentar exposição em empresas desses setores.',
                'sectors': top_sectors
            })
        
        if negative_sectors and (risk_profile != 'arrojado' or market_stance in ['bearish', 'moderately_bearish']):
            bottom_sectors = negative_sectors[:3] if len(negative_sectors) > 3 else negative_sectors
            recommendations.append({
                'type': 'sectors',
                'action': 'Evite ou reduza exposição nos setores em baixa',
                'description': f'Os setores {", ".join(bottom_sectors)} estão apresentando fraco desempenho. Considere reduzir exposição em empresas desses setores.',
                'sectors': bottom_sectors
            })
        
        # Recomendação de diversificação para perfil conservador
        if risk_profile == 'conservador':
            recommendations.append({
                'type': 'diversification',
                'action': 'Mantenha uma carteira bem diversificada',
                'description': 'Independente da tendência de mercado, mantenha uma carteira diversificada com foco em preservação de capital.'
            })
        
        # Recomendação de oportunidades para perfil arrojado
        if risk_profile == 'arrojado' and market_stance in ['bullish', 'moderately_bullish']:
            recommendations.append({
                'type': 'opportunity',
                'action': 'Busque oportunidades de crescimento',
                'description': 'Com seu perfil arrojado e o mercado favorável, busque empresas com alto potencial de crescimento e inovação.'
            })
        
        return recommendations
