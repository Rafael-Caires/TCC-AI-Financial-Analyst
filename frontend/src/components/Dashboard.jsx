import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { TrendingUp, TrendingDown, DollarSign, PieChart, AlertTriangle, Target } from 'lucide-react'

export default function Dashboard() {
  const { user } = useAuth()
  const [dashboardData, setDashboardData] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        const token = localStorage.getItem('token')
        const response = await fetch('http://localhost:5000/api/dashboard', {
          headers: {
            'Authorization': `Bearer ${token}`
          }
        })

        if (response.ok) {
          const data = await response.json()
          setDashboardData(data)
        } else {
          // Se a API não estiver disponível, usar dados simulados
          setDashboardData({
            portfolio: {
              total_value: 125430.00,
              monthly_return: 3240.00,
              monthly_return_percentage: 12.5,
              assets_count: 12,
              risk_level: 'Médio'
            },
            stocks: [
              { symbol: 'PETR4', name: 'Petrobras', price: 32.45, change: 7.6, change_value: 2.30 },
              { symbol: 'VALE3', name: 'Vale', price: 68.90, change: -1.7, change_value: -1.20 },
              { symbol: 'ITUB4', name: 'Itaú Unibanco', price: 25.80, change: 2.0, change_value: 0.50 },
              { symbol: 'BBDC4', name: 'Bradesco', price: 14.25, change: -2.1, change_value: -0.30 },
              { symbol: 'ABEV3', name: 'Ambev', price: 11.45, change: 7.5, change_value: 0.80 }
            ],
            recommendations: [
              {
                type: 'buy',
                symbol: 'ITUB4',
                title: 'ITUB4 - Comprar',
                description: 'Ação com potencial de valorização baseada em análise técnica.',
                confidence: 85
              },
              {
                type: 'diversification',
                title: 'Diversificação',
                description: 'Considere adicionar mais ativos do setor de tecnologia.',
                confidence: null
              },
              {
                type: 'alert',
                symbol: 'VALE3',
                title: 'Atenção',
                description: 'VALE3 apresenta volatilidade alta. Monitore de perto.',
                confidence: null
              }
            ]
          })
        }
      } catch (error) {
        console.error('Erro ao carregar dados do dashboard:', error)
        // Usar dados simulados em caso de erro
        setDashboardData({
          portfolio: {
            total_value: 125430.00,
            monthly_return: 3240.00,
            monthly_return_percentage: 12.5,
            assets_count: 12,
            risk_level: 'Médio'
          },
          stocks: [
            { symbol: 'PETR4', name: 'Petrobras', price: 32.45, change: 7.6, change_value: 2.30 },
            { symbol: 'VALE3', name: 'Vale', price: 68.90, change: -1.7, change_value: -1.20 },
            { symbol: 'ITUB4', name: 'Itaú Unibanco', price: 25.80, change: 2.0, change_value: 0.50 },
            { symbol: 'BBDC4', name: 'Bradesco', price: 14.25, change: -2.1, change_value: -0.30 },
            { symbol: 'ABEV3', name: 'Ambev', price: 11.45, change: 7.5, change_value: 0.80 }
          ],
          recommendations: [
            {
              type: 'buy',
              symbol: 'ITUB4',
              title: 'ITUB4 - Comprar',
              description: 'Ação com potencial de valorização baseada em análise técnica.',
              confidence: 85
            },
            {
              type: 'diversification',
              title: 'Diversificação',
              description: 'Considere adicionar mais ativos do setor de tecnologia.',
              confidence: null
            },
            {
              type: 'alert',
              symbol: 'VALE3',
              title: 'Atenção',
              description: 'VALE3 apresenta volatilidade alta. Monitore de perto.',
              confidence: null
            }
          ]
        })
      } finally {
        setLoading(false)
      }
    }

    fetchDashboardData()
  }, [])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    )
  }

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value)
  }

  const getRecommendationIcon = (type) => {
    switch (type) {
      case 'buy':
        return <TrendingUp className="w-4 h-4 text-green-600" />
      case 'diversification':
        return <Target className="w-4 h-4 text-blue-600" />
      case 'alert':
        return <AlertTriangle className="w-4 h-4 text-yellow-600" />
      default:
        return <Target className="w-4 h-4" />
    }
  }

  const getRecommendationColor = (type) => {
    switch (type) {
      case 'buy':
        return 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950'
      case 'diversification':
        return 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950'
      case 'alert':
        return 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-950'
      default:
        return 'border-gray-200 bg-gray-50 dark:border-gray-800 dark:bg-gray-950'
    }
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Bem-vindo, {user?.name}!</h1>
        <p className="text-muted-foreground">
          Aqui está um resumo do seu portfólio e do mercado financeiro.
        </p>
      </div>

      {/* Cards de estatísticas */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Patrimônio Total</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(dashboardData?.portfolio?.total_value || 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              <span className="text-green-600">+12.5% este mês</span>
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Rendimento Mensal</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(dashboardData?.portfolio?.monthly_return || 0)}
            </div>
            <p className="text-xs text-muted-foreground">
              <span className="text-green-600">+8.2% vs mês anterior</span>
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Ativos em Carteira</CardTitle>
            <PieChart className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dashboardData?.portfolio?.assets_count || 0}
            </div>
            <p className="text-xs text-muted-foreground">
              <span className="text-blue-600">Diversificado</span>
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Risco da Carteira</CardTitle>
            <AlertTriangle className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {dashboardData?.portfolio?.risk_level || 'Médio'}
            </div>
            <p className="text-xs text-muted-foreground">
              <span className="text-yellow-600">Compatível com perfil</span>
            </p>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Principais Ações */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <TrendingUp className="w-5 h-5" />
              Principais Ações
            </CardTitle>
            <CardDescription>
              Acompanhe o desempenho das principais ações do mercado
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {dashboardData?.stocks?.map((stock) => (
                <div key={stock.symbol} className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                      <span className="text-sm font-bold text-blue-600 dark:text-blue-400">
                        {stock.symbol.slice(0, 2)}
                      </span>
                    </div>
                    <div>
                      <p className="font-medium">{stock.symbol}</p>
                      <p className="text-sm text-muted-foreground">{formatCurrency(stock.price)}</p>
                    </div>
                  </div>
                  <div className="text-right">
                    <div className={`flex items-center gap-1 ${
                      stock.change >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {stock.change >= 0 ? (
                        <TrendingUp className="w-4 h-4" />
                      ) : (
                        <TrendingDown className="w-4 h-4" />
                      )}
                      <span className="font-medium">
                        {stock.change >= 0 ? '+' : ''}{stock.change.toFixed(1)}%
                      </span>
                    </div>
                    <p className="text-sm text-muted-foreground">
                      {formatCurrency(stock.change_value)}
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Recomendações */}
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center gap-2">
              <Target className="w-5 h-5" />
              Recomendações
            </CardTitle>
            <CardDescription>
              Sugestões personalizadas baseadas no seu perfil
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {dashboardData?.recommendations?.map((rec, index) => (
                <div 
                  key={index} 
                  className={`p-4 rounded-lg border ${getRecommendationColor(rec.type)}`}
                >
                  <div className="flex items-start gap-3">
                    {getRecommendationIcon(rec.type)}
                    <div className="flex-1">
                      <h4 className="font-medium">{rec.title}</h4>
                      <p className="text-sm text-muted-foreground mt-1">
                        {rec.description}
                      </p>
                      {rec.confidence && (
                        <Badge variant="secondary" className="mt-2">
                          Confiança: {rec.confidence}%
                        </Badge>
                      )}
                      {rec.type === 'diversification' && (
                        <Badge variant="outline" className="mt-2">
                          Recomendação
                        </Badge>
                      )}
                      {rec.type === 'alert' && (
                        <Badge variant="destructive" className="mt-2">
                          Alerta
                        </Badge>
                      )}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}

