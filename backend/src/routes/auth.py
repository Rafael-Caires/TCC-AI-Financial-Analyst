"""
Rotas de autenticação para o sistema

Este arquivo implementa as rotas de autenticação para o sistema de análise financeira.

Autor: Rafael Lima Caires
Data: Junho 2025
"""

from flask import Blueprint, request, jsonify, current_app
from src.models.models import User
from src.utils.auth import hash_password, check_password, generate_token, token_required, validate_input, sanitize_user_data
import json
import os

# Cria o blueprint para as rotas de autenticação
auth_bp = Blueprint('auth', __name__, url_prefix='/api/auth')

# Caminho para o arquivo de usuários (simulando um banco de dados)
USERS_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'users.json')

# Garante que o diretório de dados existe
os.makedirs(os.path.dirname(USERS_FILE), exist_ok=True)

# Função para carregar usuários do arquivo
def load_users():
    """
    Carrega usuários do arquivo JSON.
    
    Returns:
        list: Lista de usuários
    """
    if not os.path.exists(USERS_FILE):
        return []
    
    try:
        with open(USERS_FILE, 'r') as f:
            users_data = json.load(f)
        
        users = []
        for user_data in users_data:
            users.append(User.from_dict(user_data))
        
        return users
    except Exception as e:
        current_app.logger.error(f"Erro ao carregar usuários: {str(e)}")
        return []

# Função para salvar usuários no arquivo
def save_users(users):
    """
    Salva usuários no arquivo JSON.
    
    Args:
        users (list): Lista de usuários
    """
    try:
        users_data = [user.to_dict() for user in users]
        
        # Adiciona o campo password que foi removido em to_dict()
        for i, user in enumerate(users):
            users_data[i]['password'] = user.password
        
        with open(USERS_FILE, 'w') as f:
            json.dump(users_data, f, indent=2)
    except Exception as e:
        current_app.logger.error(f"Erro ao salvar usuários: {str(e)}")

# Função para encontrar um usuário pelo email
def find_user_by_email(email):
    """
    Encontra um usuário pelo email.
    
    Args:
        email (str): Email do usuário
        
    Returns:
        tuple: (usuário, índice) ou (None, -1) se não encontrado
    """
    users = load_users()
    
    for i, user in enumerate(users):
        if user.email == email:
            return user, i
    
    return None, -1

# Função para encontrar um usuário pelo ID
def find_user_by_id(user_id):
    """
    Encontra um usuário pelo ID.
    
    Args:
        user_id (int): ID do usuário
        
    Returns:
        tuple: (usuário, índice) ou (None, -1) se não encontrado
    """
    users = load_users()
    
    for i, user in enumerate(users):
        if user.id == user_id:
            return user, i
    
    return None, -1

# Rota para registro de usuário
@auth_bp.route('/register', methods=['POST'])
def register():
    """
    Registra um novo usuário.
    
    Returns:
        JSON: Resposta com status do registro
    """
    data = request.get_json()
    
    # Valida os campos obrigatórios
    valid, message = validate_input(data, ['name', 'email', 'password', 'risk_profile'])
    if not valid:
        return jsonify({'success': False, 'message': message}), 400
    
    # Verifica se o email já está em uso
    existing_user, _ = find_user_by_email(data['email'])
    if existing_user:
        return jsonify({'success': False, 'message': 'Email já cadastrado'}), 400
    
    # Verifica o perfil de risco
    if data['risk_profile'] not in ['conservador', 'moderado', 'arrojado']:
        return jsonify({'success': False, 'message': 'Perfil de risco inválido'}), 400
    
    # Carrega usuários existentes
    users = load_users()
    
    # Gera um novo ID
    new_id = 1
    if users:
        new_id = max(user.id for user in users) + 1
    
    # Cria o novo usuário
    new_user = User(
        id=new_id,
        name=data['name'],
        email=data['email'],
        password=data['password'],  # O hash será gerado no construtor
        risk_profile=data['risk_profile']
    )
    
    # Adiciona o usuário à lista
    users.append(new_user)
    
    # Salva a lista atualizada
    save_users(users)
    
    # Gera um token para o novo usuário
    token = generate_token(new_user.id)
    
    # Retorna os dados do usuário (sem a senha) e o token
    return jsonify({
        'success': True,
        'message': 'Usuário registrado com sucesso',
        'user': sanitize_user_data(new_user.to_dict()),
        'token': token
    }), 201

# Rota para login
@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Autentica um usuário.
    
    Returns:
        JSON: Resposta com token de autenticação
    """
    data = request.get_json()
    
    # Valida os campos obrigatórios
    valid, message = validate_input(data, ['email', 'password'])
    if not valid:
        return jsonify({'success': False, 'message': message}), 400
    
    # Busca o usuário pelo email
    user, _ = find_user_by_email(data['email'])
    
    # Verifica se o usuário existe e a senha está correta
    if not user or not user.check_password(data['password']):
        return jsonify({'success': False, 'message': 'Email ou senha incorretos'}), 401
    
    # Gera um token para o usuário
    token = generate_token(user.id)
    
    # Retorna os dados do usuário (sem a senha) e o token
    return jsonify({
        'success': True,
        'message': 'Login realizado com sucesso',
        'user': sanitize_user_data(user.to_dict()),
        'token': token
    }), 200

# Rota para verificar autenticação
@auth_bp.route('/verify', methods=['GET'])
@token_required
def verify(user_id):
    """
    Verifica se o token é válido e retorna os dados do usuário.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Resposta com dados do usuário
    """
    # Busca o usuário pelo ID
    user, _ = find_user_by_id(user_id)
    
    if not user:
        return jsonify({'authenticated': False, 'message': 'Usuário não encontrado'}), 404
    
    # Retorna os dados do usuário (sem a senha)
    return jsonify({
        'authenticated': True,
        'message': 'Token válido',
        'user': sanitize_user_data(user.to_dict())
    }), 200

# Rota para atualizar perfil
@auth_bp.route('/update-profile', methods=['PUT'])
@token_required
def update_profile(user_id):
    """
    Atualiza o perfil do usuário.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Resposta com status da atualização
    """
    data = request.get_json()
    
    # Busca o usuário pelo ID
    user, index = find_user_by_id(user_id)
    
    if not user:
        return jsonify({'success': False, 'message': 'Usuário não encontrado'}), 404
    
    # Carrega todos os usuários
    users = load_users()
    
    # Atualiza os campos permitidos
    if 'name' in data:
        user.name = data['name']
    
    if 'risk_profile' in data:
        if data['risk_profile'] in ['conservador', 'moderado', 'arrojado']:
            user.risk_profile = data['risk_profile']
        else:
            return jsonify({'success': False, 'message': 'Perfil de risco inválido'}), 400
    
    # Atualiza a senha se fornecida
    if 'password' in data and data['password']:
        # Verifica a senha atual se fornecida
        if 'current_password' in data and data['current_password']:
            if not user.check_password(data['current_password']):
                return jsonify({'success': False, 'message': 'Senha atual incorreta'}), 400
        
        # Atualiza a senha
        user.password = hash_password(data['password'])
    
    # Atualiza o usuário na lista
    users[index] = user
    
    # Salva a lista atualizada
    save_users(users)
    
    # Retorna os dados atualizados do usuário (sem a senha)
    return jsonify({
        'success': True,
        'message': 'Perfil atualizado com sucesso',
        'user': sanitize_user_data(user.to_dict())
    }), 200

# Rota para excluir conta
@auth_bp.route('/delete-account', methods=['DELETE'])
@token_required
def delete_account(user_id):
    """
    Exclui a conta do usuário.
    
    Args:
        user_id (int): ID do usuário (injetado pelo decorator token_required)
        
    Returns:
        JSON: Resposta com status da exclusão
    """
    # Busca o usuário pelo ID
    user, index = find_user_by_id(user_id)
    
    if not user:
        return jsonify({'success': False, 'message': 'Usuário não encontrado'}), 404
    
    # Carrega todos os usuários
    users = load_users()
    
    # Remove o usuário da lista
    users.pop(index)
    
    # Salva a lista atualizada
    save_users(users)
    
    return jsonify({
        'success': True,
        'message': 'Conta excluída com sucesso'
    }), 200
