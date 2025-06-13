"""
Modelos de dados para o sistema

Este arquivo define os modelos de dados para o sistema de análise financeira.
Estes modelos representam as entidades principais do sistema.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash

class User:
    """
    Modelo para representar um usuário do sistema.
    """
    
    def __init__(self, id=None, name=None, email=None, password=None, risk_profile=None, created_at=None):
        """
        Inicializa um novo usuário.
        
        Args:
            id (int): ID do usuário
            name (str): Nome do usuário
            email (str): Email do usuário
            password (str): Senha do usuário (será armazenada como hash)
            risk_profile (str): Perfil de risco do usuário ('conservador', 'moderado', 'arrojado')
            created_at (datetime): Data de criação do usuário
        """
        self.id = id
        self.name = name
        self.email = email
        # Se a senha já é um hash (começa com pbkdf2), não faz hash novamente
        if password and password.startswith('pbkdf2'):
            self.password = password
        else:
            self.password = generate_password_hash(password) if password else None
        self.risk_profile = risk_profile
        self.created_at = created_at or datetime.now()
    
    def check_password(self, password):
        """
        Verifica se a senha fornecida corresponde ao hash armazenado.
        
        Args:
            password (str): Senha em texto plano para verificar
            
        Returns:
            bool: True se a senha estiver correta, False caso contrário
        """
        return check_password_hash(self.password, password)
    
    def to_dict(self):
        """
        Converte o usuário para um dicionário.
        
        Returns:
            dict: Representação do usuário como dicionário
        """
        return {
            'id': self.id,
            'name': self.name,
            'email': self.email,
            'risk_profile': self.risk_profile,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Cria um usuário a partir de um dicionário.
        
        Args:
            data (dict): Dicionário com dados do usuário
            
        Returns:
            User: Instância de usuário
        """
        return cls(
            id=data.get('id'),
            name=data.get('name'),
            email=data.get('email'),
            password=data.get('password'),
            risk_profile=data.get('risk_profile'),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else None
        )


class Portfolio:
    """
    Modelo para representar um portfólio de investimentos.
    """
    
    def __init__(self, id=None, user_id=None, name=None, description=None, created_at=None):
        """
        Inicializa um novo portfólio.
        
        Args:
            id (int): ID do portfólio
            user_id (int): ID do usuário proprietário
            name (str): Nome do portfólio
            description (str): Descrição do portfólio
            created_at (datetime): Data de criação do portfólio
        """
        self.id = id
        self.user_id = user_id
        self.name = name
        self.description = description
        self.created_at = created_at or datetime.now()
        self.assets = []
    
    def add_asset(self, asset):
        """
        Adiciona um ativo ao portfólio.
        
        Args:
            asset (PortfolioAsset): Ativo a ser adicionado
        """
        self.assets.append(asset)
    
    def remove_asset(self, asset_id):
        """
        Remove um ativo do portfólio.
        
        Args:
            asset_id (int): ID do ativo a ser removido
            
        Returns:
            bool: True se o ativo foi removido, False caso contrário
        """
        for i, asset in enumerate(self.assets):
            if asset.id == asset_id:
                self.assets.pop(i)
                return True
        return False
    
    def to_dict(self):
        """
        Converte o portfólio para um dicionário.
        
        Returns:
            dict: Representação do portfólio como dicionário
        """
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'description': self.description,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'assets': [asset.to_dict() for asset in self.assets]
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Cria um portfólio a partir de um dicionário.
        
        Args:
            data (dict): Dicionário com dados do portfólio
            
        Returns:
            Portfolio: Instância de portfólio
        """
        portfolio = cls(
            id=data.get('id'),
            user_id=data.get('user_id'),
            name=data.get('name'),
            description=data.get('description'),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else None
        )
        
        if 'assets' in data:
            for asset_data in data['assets']:
                portfolio.add_asset(PortfolioAsset.from_dict(asset_data))
        
        return portfolio


class PortfolioAsset:
    """
    Modelo para representar um ativo em um portfólio.
    """
    
    def __init__(self, id=None, portfolio_id=None, ticker=None, quantity=None, purchase_price=None, purchase_date=None):
        """
        Inicializa um novo ativo de portfólio.
        
        Args:
            id (int): ID do ativo
            portfolio_id (int): ID do portfólio
            ticker (str): Símbolo do ativo
            quantity (float): Quantidade do ativo
            purchase_price (float): Preço de compra
            purchase_date (datetime): Data de compra
        """
        self.id = id
        self.portfolio_id = portfolio_id
        self.ticker = ticker
        self.quantity = quantity
        self.purchase_price = purchase_price
        self.purchase_date = purchase_date or datetime.now()
    
    def to_dict(self):
        """
        Converte o ativo para um dicionário.
        
        Returns:
            dict: Representação do ativo como dicionário
        """
        return {
            'id': self.id,
            'portfolio_id': self.portfolio_id,
            'ticker': self.ticker,
            'quantity': self.quantity,
            'purchase_price': self.purchase_price,
            'purchase_date': self.purchase_date.isoformat() if self.purchase_date else None
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Cria um ativo a partir de um dicionário.
        
        Args:
            data (dict): Dicionário com dados do ativo
            
        Returns:
            PortfolioAsset: Instância de ativo
        """
        return cls(
            id=data.get('id'),
            portfolio_id=data.get('portfolio_id'),
            ticker=data.get('ticker'),
            quantity=data.get('quantity'),
            purchase_price=data.get('purchase_price'),
            purchase_date=datetime.fromisoformat(data['purchase_date']) if 'purchase_date' in data else None
        )


class Watchlist:
    """
    Modelo para representar uma lista de observação de ativos.
    """
    
    def __init__(self, id=None, user_id=None, name=None, created_at=None):
        """
        Inicializa uma nova lista de observação.
        
        Args:
            id (int): ID da lista
            user_id (int): ID do usuário proprietário
            name (str): Nome da lista
            created_at (datetime): Data de criação da lista
        """
        self.id = id
        self.user_id = user_id
        self.name = name
        self.created_at = created_at or datetime.now()
        self.tickers = []
    
    def add_ticker(self, ticker):
        """
        Adiciona um ticker à lista de observação.
        
        Args:
            ticker (str): Ticker a ser adicionado
            
        Returns:
            bool: True se o ticker foi adicionado, False se já existia
        """
        if ticker not in self.tickers:
            self.tickers.append(ticker)
            return True
        return False
    
    def remove_ticker(self, ticker):
        """
        Remove um ticker da lista de observação.
        
        Args:
            ticker (str): Ticker a ser removido
            
        Returns:
            bool: True se o ticker foi removido, False caso contrário
        """
        if ticker in self.tickers:
            self.tickers.remove(ticker)
            return True
        return False
    
    def to_dict(self):
        """
        Converte a lista de observação para um dicionário.
        
        Returns:
            dict: Representação da lista como dicionário
        """
        return {
            'id': self.id,
            'user_id': self.user_id,
            'name': self.name,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'tickers': self.tickers
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Cria uma lista de observação a partir de um dicionário.
        
        Args:
            data (dict): Dicionário com dados da lista
            
        Returns:
            Watchlist: Instância de lista de observação
        """
        watchlist = cls(
            id=data.get('id'),
            user_id=data.get('user_id'),
            name=data.get('name'),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else None
        )
        
        if 'tickers' in data:
            watchlist.tickers = data['tickers']
        
        return watchlist


class Prediction:
    """
    Modelo para representar uma previsão de preço.
    """
    
    def __init__(self, id=None, ticker=None, model_type=None, prediction_date=None, data=None):
        """
        Inicializa uma nova previsão.
        
        Args:
            id (int): ID da previsão
            ticker (str): Símbolo do ativo
            model_type (str): Tipo de modelo usado ('lstm', 'random_forest', 'lightgbm', 'ensemble')
            prediction_date (datetime): Data da previsão
            data (dict): Dados da previsão
        """
        self.id = id
        self.ticker = ticker
        self.model_type = model_type
        self.prediction_date = prediction_date or datetime.now()
        self.data = data or {}
    
    def to_dict(self):
        """
        Converte a previsão para um dicionário.
        
        Returns:
            dict: Representação da previsão como dicionário
        """
        return {
            'id': self.id,
            'ticker': self.ticker,
            'model_type': self.model_type,
            'prediction_date': self.prediction_date.isoformat() if self.prediction_date else None,
            'data': self.data
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Cria uma previsão a partir de um dicionário.
        
        Args:
            data (dict): Dicionário com dados da previsão
            
        Returns:
            Prediction: Instância de previsão
        """
        return cls(
            id=data.get('id'),
            ticker=data.get('ticker'),
            model_type=data.get('model_type'),
            prediction_date=datetime.fromisoformat(data['prediction_date']) if 'prediction_date' in data else None,
            data=data.get('data', {})
        )


class Recommendation:
    """
    Modelo para representar uma recomendação de investimento.
    """
    
    def __init__(self, id=None, user_id=None, ticker=None, action=None, confidence=None, reason=None, created_at=None):
        """
        Inicializa uma nova recomendação.
        
        Args:
            id (int): ID da recomendação
            user_id (int): ID do usuário
            ticker (str): Símbolo do ativo
            action (str): Ação recomendada ('comprar', 'vender', 'manter')
            confidence (float): Nível de confiança (0-1)
            reason (str): Justificativa da recomendação
            created_at (datetime): Data de criação da recomendação
        """
        self.id = id
        self.user_id = user_id
        self.ticker = ticker
        self.action = action
        self.confidence = confidence
        self.reason = reason
        self.created_at = created_at or datetime.now()
    
    def to_dict(self):
        """
        Converte a recomendação para um dicionário.
        
        Returns:
            dict: Representação da recomendação como dicionário
        """
        return {
            'id': self.id,
            'user_id': self.user_id,
            'ticker': self.ticker,
            'action': self.action,
            'confidence': self.confidence,
            'reason': self.reason,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }
    
    @classmethod
    def from_dict(cls, data):
        """
        Cria uma recomendação a partir de um dicionário.
        
        Args:
            data (dict): Dicionário com dados da recomendação
            
        Returns:
            Recommendation: Instância de recomendação
        """
        return cls(
            id=data.get('id'),
            user_id=data.get('user_id'),
            ticker=data.get('ticker'),
            action=data.get('action'),
            confidence=data.get('confidence'),
            reason=data.get('reason'),
            created_at=datetime.fromisoformat(data['created_at']) if 'created_at' in data else None
        )
