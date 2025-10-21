"""
Sistema Inteligente para Análise Preditiva e Recomendação de Estratégias para o Mercado de Renda Variável
Arquivo principal da aplicação Flask (Backend)

Este arquivo configura e inicializa a aplicação Flask, registra as rotas
e define o comportamento principal da API.

Funcionalidades principais:
- Autenticação de usuários
- Previsões com ensemble de modelos de ML
- Análise de sentimentos de notícias
- Análise quantitativa de risco
- Sistema de recomendação híbrido
- Backtesting de estratégias
- Stress testing de portfólios

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import os
import sys
from flask import Flask, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# Configuração para importações absolutas
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Importação das rotas
from src.routes.auth import auth_bp
from src.routes.stocks import stocks_bp
from src.routes.predictions import predictions_bp
from src.routes.recommendations import recommendations_bp
from src.routes.dashboard import dashboard_bp
from src.routes.ai_routes import ai_bp
from src.routes.ai_analysis import ai_analysis_bp
from src.routes.portfolio import portfolio_bp

# Carrega variáveis de ambiente
load_dotenv()

def create_app(test_config=None):
    """
    Função factory para criar e configurar a aplicação Flask
    """
    # Cria e configura a aplicação
    app = Flask(__name__, instance_relative_config=True)
    
    # Configuração básica
    app.config.from_mapping(
        SECRET_KEY=os.environ.get('SECRET_KEY', 'dev'),
        # Configuração do banco de dados (descomente se necessário)
        # SQLALCHEMY_DATABASE_URI=f"mysql+pymysql://{os.getenv('DB_USERNAME', 'root')}:{os.getenv('DB_PASSWORD', 'password')}@{os.getenv('DB_HOST', 'localhost')}:{os.getenv('DB_PORT', '3306')}/{os.getenv('DB_NAME', 'finance_db')}",
        # SQLALCHEMY_TRACK_MODIFICATIONS=False,
    )
    
    # Habilita CORS para permitir requisições do frontend
    CORS(app, resources={
        r"/api/*": {
            "origins": ["http://localhost:3000", "http://localhost:5173", "http://localhost:5174", "http://127.0.0.1:5173", "http://127.0.0.1:3000", "http://127.0.0.1:5174"],
            "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            "allow_headers": ["Content-Type", "Authorization"]
        }
    })
    
    # Registra os blueprints (rotas)
    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(stocks_bp, url_prefix='/api/stocks')
    app.register_blueprint(predictions_bp, url_prefix='/api/predictions')
    app.register_blueprint(recommendations_bp, url_prefix='/api/recommendations')
    app.register_blueprint(dashboard_bp, url_prefix='/api/dashboard')
    app.register_blueprint(ai_bp)  # Rotas de IA avançadas
    app.register_blueprint(ai_analysis_bp)  # Nova análise com IA
    app.register_blueprint(portfolio_bp)  # Rotas avançadas de portfólio
    
    # Rota de teste/status
    @app.route('/api/status')
    def status():
        return jsonify({
            'status': 'online',
            'message': 'Sistema Financeiro com IA - Backend funcionando',
            'version': '2.0.0',
            'features': [
                'Autenticação JWT',
                'Previsões com Ensemble de Modelos',
                'Análise de Sentimentos',
                'Análise Quantitativa de Risco',
                'Sistema de Recomendação Híbrido',
                'Backtesting de Estratégias',
                'Stress Testing de Portfólios',
                'Análise Técnica Avançada'
            ]
        })
    
    # Rota para portfólio (compatibilidade)
    @app.route('/api/portfolio')
    def portfolio_redirect():
        from src.routes.dashboard import get_portfolio
        return get_portfolio()
    
    # Rota para perfil (compatibilidade)
    @app.route('/api/profile', methods=['PUT'])
    def profile_redirect():
        from src.routes.dashboard import update_profile
        return update_profile()
    
    return app

# Cria a aplicação
app = create_app()

if __name__ == '__main__':
    # Executa a aplicação em modo de desenvolvimento
    port = int(os.environ.get('PORT', 5000))
    
    print(f"""

 Servidor rodando em: http://0.0.0.0:{port} 

    """)
    
    app.run(host='0.0.0.0', port=port, debug=True)
