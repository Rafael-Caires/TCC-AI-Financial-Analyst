import { useState } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { Avatar, AvatarFallback } from './ui/avatar'
import { Badge } from './ui/badge'
import { User, Mail, Shield, Save, Eye, EyeOff } from 'lucide-react'

export default function Profile() {
  const { user, logout } = useAuth()
  const [formData, setFormData] = useState({
    name: user?.name || '',
    email: user?.email || '',
    currentPassword: '',
    newPassword: '',
    confirmPassword: '',
    riskProfile: user?.risk_profile || ''
  })
  const [showCurrentPassword, setShowCurrentPassword] = useState(false)
  const [showNewPassword, setShowNewPassword] = useState(false)
  const [showConfirmPassword, setShowConfirmPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [message, setMessage] = useState('')

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value
    })
  }

  const handleRiskProfileChange = (value) => {
    setFormData({
      ...formData,
      riskProfile: value
    })
  }

  const handleSubmit = async (e) => {
    e.preventDefault()
    setLoading(true)
    setMessage('')

    try {
      const token = localStorage.getItem('token')
      const updateData = {
        name: formData.name,
        risk_profile: formData.riskProfile
      }

      // Se está tentando alterar a senha
      if (formData.newPassword) {
        if (formData.newPassword !== formData.confirmPassword) {
          setMessage('As senhas não coincidem')
          setLoading(false)
          return
        }
        if (formData.newPassword.length < 6) {
          setMessage('A nova senha deve ter pelo menos 6 caracteres')
          setLoading(false)
          return
        }
        updateData.current_password = formData.currentPassword
        updateData.new_password = formData.newPassword
      }

      const response = await fetch('http://localhost:5000/api/profile', {
        method: 'PUT',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify(updateData)
      })

      if (response.ok) {
        const data = await response.json()
        // Atualizar dados do usuário no localStorage
        const updatedUser = { ...user, ...data.user }
        localStorage.setItem('user', JSON.stringify(updatedUser))
        setMessage('Perfil atualizado com sucesso!')
        
        // Limpar campos de senha
        setFormData({
          ...formData,
          currentPassword: '',
          newPassword: '',
          confirmPassword: ''
        })
      } else {
        const data = await response.json()
        setMessage(data.message || 'Erro ao atualizar perfil')
      }
    } catch (error) {
      console.error('Erro ao atualizar perfil:', error)
      setMessage('Erro de conexão com o servidor')
    } finally {
      setLoading(false)
    }
  }

  const getUserInitials = (name) => {
    return name
      .split(' ')
      .map(word => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2)
  }

  const getRiskProfileColor = (profile) => {
    switch (profile) {
      case 'conservador':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'moderado':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      case 'arrojado':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const getRiskProfileLabel = (profile) => {
    switch (profile) {
      case 'conservador':
        return 'Conservador'
      case 'moderado':
        return 'Moderado'
      case 'arrojado':
        return 'Arrojado'
      default:
        return profile
    }
  }

  const riskProfiles = [
    { value: 'conservador', label: 'Conservador', description: 'Prefere investimentos seguros' },
    { value: 'moderado', label: 'Moderado', description: 'Equilibra risco e retorno' },
    { value: 'arrojado', label: 'Arrojado', description: 'Aceita riscos maiores por retornos altos' }
  ]

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Perfil</h1>
        <p className="text-muted-foreground">
          Gerencie suas informações pessoais e preferências
        </p>
      </div>

      <div className="grid gap-6 md:grid-cols-3">
        {/* Informações do Usuário */}
        <Card>
          <CardHeader className="text-center">
            <Avatar className="w-20 h-20 mx-auto mb-4">
              <AvatarFallback className="text-lg">
                {getUserInitials(user?.name || 'U')}
              </AvatarFallback>
            </Avatar>
            <CardTitle>{user?.name}</CardTitle>
            <CardDescription>{user?.email}</CardDescription>
          </CardHeader>
          <CardContent className="text-center space-y-4">
            <div>
              <p className="text-sm font-medium mb-2">Perfil de Risco</p>
              <Badge className={getRiskProfileColor(user?.risk_profile)}>
                {getRiskProfileLabel(user?.risk_profile)}
              </Badge>
            </div>
            <div>
              <p className="text-sm font-medium mb-2">Membro desde</p>
              <p className="text-sm text-muted-foreground">
                {user?.created_at ? new Date(user.created_at).toLocaleDateString('pt-BR') : 'N/A'}
              </p>
            </div>
          </CardContent>
        </Card>

        {/* Formulário de Edição */}
        <Card className="md:col-span-2">
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <User className="w-5 h-5" />
              Editar Perfil
            </CardTitle>
            <CardDescription>
              Atualize suas informações pessoais
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-6">
              {message && (
                <div className={`p-3 text-sm rounded-md ${
                  message.includes('sucesso') 
                    ? 'text-green-600 bg-green-50 border border-green-200' 
                    : 'text-red-600 bg-red-50 border border-red-200'
                }`}>
                  {message}
                </div>
              )}

              {/* Informações Básicas */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Informações Básicas</h3>
                
                <div className="space-y-2">
                  <Label htmlFor="name">Nome Completo</Label>
                  <Input
                    id="name"
                    name="name"
                    type="text"
                    value={formData.name}
                    onChange={handleChange}
                    required
                  />
                </div>

                <div className="space-y-2">
                  <Label htmlFor="email">Email</Label>
                  <Input
                    id="email"
                    name="email"
                    type="email"
                    value={formData.email}
                    disabled
                    className="bg-muted"
                  />
                  <p className="text-xs text-muted-foreground">
                    O email não pode ser alterado
                  </p>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="riskProfile">Perfil de Risco</Label>
                  <Select value={formData.riskProfile} onValueChange={handleRiskProfileChange}>
                    <SelectTrigger>
                      <SelectValue placeholder="Selecione seu perfil" />
                    </SelectTrigger>
                    <SelectContent>
                      {riskProfiles.map((profile) => (
                        <SelectItem key={profile.value} value={profile.value}>
                          <div className="flex flex-col">
                            <span className="font-medium">{profile.label}</span>
                            <span className="text-sm text-muted-foreground">{profile.description}</span>
                          </div>
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
              </div>

              {/* Alteração de Senha */}
              <div className="space-y-4">
                <h3 className="text-lg font-medium">Alterar Senha</h3>
                <p className="text-sm text-muted-foreground">
                  Deixe em branco se não quiser alterar a senha
                </p>

                <div className="space-y-2">
                  <Label htmlFor="currentPassword">Senha Atual</Label>
                  <div className="relative">
                    <Input
                      id="currentPassword"
                      name="currentPassword"
                      type={showCurrentPassword ? 'text' : 'password'}
                      value={formData.currentPassword}
                      onChange={handleChange}
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                      onClick={() => setShowCurrentPassword(!showCurrentPassword)}
                    >
                      {showCurrentPassword ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="newPassword">Nova Senha</Label>
                  <div className="relative">
                    <Input
                      id="newPassword"
                      name="newPassword"
                      type={showNewPassword ? 'text' : 'password'}
                      value={formData.newPassword}
                      onChange={handleChange}
                      placeholder="Mínimo 6 caracteres"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                      onClick={() => setShowNewPassword(!showNewPassword)}
                    >
                      {showNewPassword ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>

                <div className="space-y-2">
                  <Label htmlFor="confirmPassword">Confirmar Nova Senha</Label>
                  <div className="relative">
                    <Input
                      id="confirmPassword"
                      name="confirmPassword"
                      type={showConfirmPassword ? 'text' : 'password'}
                      value={formData.confirmPassword}
                      onChange={handleChange}
                      placeholder="Confirme a nova senha"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    >
                      {showConfirmPassword ? (
                        <EyeOff className="h-4 w-4" />
                      ) : (
                        <Eye className="h-4 w-4" />
                      )}
                    </Button>
                  </div>
                </div>
              </div>

              <div className="flex gap-4">
                <Button type="submit" disabled={loading}>
                  <Save className="w-4 h-4 mr-2" />
                  {loading ? 'Salvando...' : 'Salvar Alterações'}
                </Button>
                <Button type="button" variant="outline" onClick={logout}>
                  Sair da Conta
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      </div>

      {/* Informações de Segurança */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Shield className="w-5 h-5" />
            Segurança da Conta
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-4">
            <div className="flex items-center justify-between p-4 border rounded-lg">
              <div>
                <h4 className="font-medium">Autenticação de Dois Fatores</h4>
                <p className="text-sm text-muted-foreground">
                  Adicione uma camada extra de segurança à sua conta
                </p>
              </div>
              <Button variant="outline" disabled>
                Em Breve
              </Button>
            </div>
            
            <div className="flex items-center justify-between p-4 border rounded-lg">
              <div>
                <h4 className="font-medium">Histórico de Login</h4>
                <p className="text-sm text-muted-foreground">
                  Veja quando e onde você fez login na sua conta
                </p>
              </div>
              <Button variant="outline" disabled>
                Em Breve
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

