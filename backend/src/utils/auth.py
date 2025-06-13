"""
Módulo de autenticação e segurança

Este arquivo implementa funções para autenticação, geração e validação de tokens JWT,
e segurança da API.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

import jwt
import datetime
import os
from functools import wraps
from flask import request, jsonify, current_app
from werkzeug.security import generate_password_hash, check_password_hash

# Função para gerar hash de senha
def hash_password(password):
    """
    Gera um hash seguro para a senha fornecida.
    
    Args:
        password (str): Senha em texto plano
        
    Returns:
        str: Hash da senha
    """
    return generate_password_hash(password, method='pbkdf2:sha256', salt_length=16)

# Função para verificar senha
def check_password(hashed_password, password):
    """
    Verifica se a senha fornecida corresponde ao hash armazenado.
    
    Args:
        hashed_password (str): Hash da senha armazenada
        password (str): Senha em texto plano para verificar
        
    Returns:
        bool: True se a senha estiver correta, False caso contrário
    """
    return check_password_hash(hashed_password, password)

# Função para gerar token JWT
def generate_token(user_id, expiration=24):
    """
    Gera um token JWT para o usuário.
    
    Args:
        user_id (int): ID do usuário
        expiration (int): Tempo de expiração em horas
        
    Returns:
        str: Token JWT
    """
    payload = {
        'exp': datetime.datetime.utcnow() + datetime.timedelta(hours=expiration),
        'iat': datetime.datetime.utcnow(),
        'sub': user_id
    }
    
    return jwt.encode(
        payload,
        current_app.config.get('SECRET_KEY', os.getenv('SECRET_KEY', 'dev_key')),
        algorithm='HS256'
    )

# Função para decodificar token JWT
def decode_token(token):
    """
    Decodifica um token JWT.
    
    Args:
        token (str): Token JWT
        
    Returns:
        dict: Payload do token decodificado
        
    Raises:
        jwt.ExpiredSignatureError: Se o token expirou
        jwt.InvalidTokenError: Se o token é inválido
    """
    return jwt.decode(
        token,
        current_app.config.get('SECRET_KEY', os.getenv('SECRET_KEY', 'dev_key')),
        algorithms=['HS256']
    )

# Decorator para rotas que requerem autenticação
def token_required(f):
    """
    Decorator para proteger rotas que requerem autenticação.
    
    Args:
        f (function): Função da rota a ser protegida
        
    Returns:
        function: Função decorada que verifica o token antes de executar a rota
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        
        # Verifica se o token está no cabeçalho Authorization
        if 'Authorization' in request.headers:
            auth_header = request.headers['Authorization']
            if auth_header.startswith('Bearer '):
                token = auth_header.split(' ')[1]
        
        if not token:
            return jsonify({
                'message': 'Token de autenticação não fornecido',
                'authenticated': False
            }), 401
        
        try:
            # Decodifica o token
            payload = decode_token(token)
            user_id = payload['sub']
            
            # Adiciona o ID do usuário aos argumentos da função
            kwargs['user_id'] = user_id
            
        except jwt.ExpiredSignatureError:
            return jsonify({
                'message': 'Token de autenticação expirado',
                'authenticated': False
            }), 401
        except jwt.InvalidTokenError:
            return jsonify({
                'message': 'Token de autenticação inválido',
                'authenticated': False
            }), 401
        
        return f(*args, **kwargs)
    
    return decorated

# Função para validar dados de entrada
def validate_input(data, required_fields):
    """
    Valida se os campos obrigatórios estão presentes nos dados.
    
    Args:
        data (dict): Dados a serem validados
        required_fields (list): Lista de campos obrigatórios
        
    Returns:
        tuple: (bool, str) indicando se a validação passou e a mensagem de erro
    """
    missing_fields = [field for field in required_fields if field not in data]
    
    if missing_fields:
        return False, f"Campos obrigatórios ausentes: {', '.join(missing_fields)}"
    
    return True, ""

# Função para sanitizar dados de saída
def sanitize_user_data(user_data):
    """
    Remove campos sensíveis dos dados do usuário.
    
    Args:
        user_data (dict): Dados do usuário
        
    Returns:
        dict: Dados do usuário sem campos sensíveis
    """
    # Cria uma cópia para não modificar o original
    sanitized = user_data.copy()
    
    # Remove campos sensíveis
    if 'password' in sanitized:
        del sanitized['password']
    
    return sanitized
