"""
Rotas para dashboard e portfólio
"""

from flask import Blueprint, jsonify, request
from src.utils.auth import token_required

dashboard_bp = Blueprint('dashboard', __name__)

@dashboard_bp.route('', methods=['GET'])
@dashboard_bp.route('/', methods=['GET'])
def get_dashboard_data():
    """
    Retorna dados do dashboard para o usuário logado
    """
    try:
        # Dados simulados do dashboard
        dashboard_data = {
            'portfolio': {
                'total_value': 125430.00,
                'monthly_return': 3240.00,
                'monthly_return_percentage': 12.5,
                'assets_count': 12,
                'risk_level': 'Médio'
            },
            'stocks': [
                { 'symbol': 'PETR4', 'name': 'Petrobras', 'price': 32.45, 'change': 7.6, 'change_value': 2.30 },
                { 'symbol': 'VALE3', 'name': 'Vale', 'price': 68.90, 'change': -1.7, 'change_value': -1.20 },
                { 'symbol': 'ITUB4', 'name': 'Itaú Unibanco', 'price': 25.80, 'change': 2.0, 'change_value': 0.50 },
                { 'symbol': 'BBDC4', 'name': 'Bradesco', 'price': 14.25, 'change': -2.1, 'change_value': -0.30 },
                { 'symbol': 'ABEV3', 'name': 'Ambev', 'price': 11.45, 'change': 7.5, 'change_value': 0.80 }
            ],
            'recommendations': [
                {
                    'type': 'buy',
                    'symbol': 'ITUB4',
                    'title': 'ITUB4 - Comprar',
                    'description': 'Ação com potencial de valorização baseada em análise técnica.',
                    'confidence': 85
                },
                {
                    'type': 'diversification',
                    'title': 'Diversificação',
                    'description': 'Considere adicionar mais ativos do setor de tecnologia.',
                    'confidence': None
                },
                {
                    'type': 'alert',
                    'symbol': 'VALE3',
                    'title': 'Atenção',
                    'description': 'VALE3 apresenta volatilidade alta. Monitore de perto.',
                    'confidence': None
                }
            ]
        }
        
        return jsonify(dashboard_data), 200
        
    except Exception as e:
        return jsonify({'message': f'Erro interno do servidor: {str(e)}'}), 500

@dashboard_bp.route('/portfolio', methods=['GET'])
@token_required
def get_portfolio(current_user):
    """
    Retorna dados do portfólio do usuário
    """
    try:
        # Dados simulados do portfólio
        portfolio_data = {
            'total_value': 125430.00,
            'total_invested': 110000.00,
            'total_return': 15430.00,
            'return_percentage': 14.03,
            'assets': [
                {
                    'symbol': 'PETR4',
                    'name': 'Petrobras',
                    'quantity': 500,
                    'avg_price': 28.50,
                    'current_price': 32.45,
                    'total_invested': 14250.00,
                    'current_value': 16225.00,
                    'return_value': 1975.00,
                    'return_percentage': 13.86,
                    'allocation': 12.9
                },
                {
                    'symbol': 'VALE3',
                    'name': 'Vale',
                    'quantity': 200,
                    'avg_price': 65.00,
                    'current_price': 68.90,
                    'total_invested': 13000.00,
                    'current_value': 13780.00,
                    'return_value': 780.00,
                    'return_percentage': 6.00,
                    'allocation': 11.0
                },
                {
                    'symbol': 'ITUB4',
                    'name': 'Itaú Unibanco',
                    'quantity': 800,
                    'avg_price': 24.00,
                    'current_price': 25.80,
                    'total_invested': 19200.00,
                    'current_value': 20640.00,
                    'return_value': 1440.00,
                    'return_percentage': 7.50,
                    'allocation': 16.5
                },
                {
                    'symbol': 'BBDC4',
                    'name': 'Bradesco',
                    'quantity': 1000,
                    'avg_price': 15.50,
                    'current_price': 14.25,
                    'total_invested': 15500.00,
                    'current_value': 14250.00,
                    'return_value': -1250.00,
                    'return_percentage': -8.06,
                    'allocation': 11.4
                },
                {
                    'symbol': 'ABEV3',
                    'name': 'Ambev',
                    'quantity': 1500,
                    'avg_price': 10.80,
                    'current_price': 11.45,
                    'total_invested': 16200.00,
                    'current_value': 17175.00,
                    'return_value': 975.00,
                    'return_percentage': 6.02,
                    'allocation': 13.7
                },
                {
                    'symbol': 'WEGE3',
                    'name': 'WEG',
                    'quantity': 300,
                    'avg_price': 42.00,
                    'current_price': 45.30,
                    'total_invested': 12600.00,
                    'current_value': 13590.00,
                    'return_value': 990.00,
                    'return_percentage': 7.86,
                    'allocation': 10.8
                }
            ],
            'allocation_by_sector': [
                { 'sector': 'Financeiro', 'percentage': 27.9, 'value': 34890.00 },
                { 'sector': 'Petróleo e Gás', 'percentage': 12.9, 'value': 16225.00 },
                { 'sector': 'Mineração', 'percentage': 11.0, 'value': 13780.00 },
                { 'sector': 'Bebidas', 'percentage': 13.7, 'value': 17175.00 },
                { 'sector': 'Industrial', 'percentage': 10.8, 'value': 13590.00 },
                { 'sector': 'Outros', 'percentage': 23.7, 'value': 29770.00 }
            ]
        }
        
        return jsonify(portfolio_data), 200
        
    except Exception as e:
        return jsonify({'message': f'Erro interno do servidor: {str(e)}'}), 500

@dashboard_bp.route('/profile', methods=['PUT'])
@token_required
def update_profile(current_user):
    """
    Atualiza o perfil do usuário
    """
    try:
        data = request.get_json()
        
        # Validações básicas
        if 'name' in data and not data['name'].strip():
            return jsonify({'message': 'Nome não pode estar vazio'}), 400
            
        if 'risk_profile' in data and data['risk_profile'] not in ['conservador', 'moderado', 'arrojado']:
            return jsonify({'message': 'Perfil de risco inválido'}), 400
        
        # Simular atualização do perfil
        # Em um sistema real, aqui você atualizaria o banco de dados
        updated_user = {
            'id': current_user['id'],
            'name': data.get('name', current_user['name']),
            'email': current_user['email'],
            'risk_profile': data.get('risk_profile', current_user['risk_profile']),
            'created_at': current_user.get('created_at')
        }
        
        return jsonify({
            'message': 'Perfil atualizado com sucesso',
            'user': updated_user
        }), 200
        
    except Exception as e:
        return jsonify({'message': f'Erro interno do servidor: {str(e)}'}), 500

