import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Progress } from './ui/progress'
import { Briefcase, TrendingUp, TrendingDown, PieChart, Plus, Minus } from 'lucide-react'

export default function Portfolio() {
  const [portfolio, setPortfolio] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    fetchPortfolio()
  }, [])

  const fetchPortfolio = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('token')
      const response = await fetch('http://localhost:5000/api/portfolio', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setPortfolio(data)
      } else {
        // Dados simulados se a API não estiver disponível
        setPortfolio({
          total_value: 125430.00,
          total_invested: 110000.00,
          total_return: 15430.00,
          return_percentage: 14.03,
          assets: [
            {
              symbol: 'PETR4',
              name: 'Petrobras',
              quantity: 500,
              avg_price: 28.50,
              current_price: 32.45,
              total_invested: 14250.00,
              current_value: 16225.00,
              return_value: 1975.00,
              return_percentage: 13.86,
              allocation: 12.9
            },
            {
              symbol: 'VALE3',
              name: 'Vale',
              quantity: 200,
              avg_price: 65.00,
              current_price: 68.90,
              total_invested: 13000.00,
              current_value: 13780.00,
              return_value: 780.00,
              return_percentage: 6.00,
              allocation: 11.0
            },
            {
              symbol: 'ITUB4',
              name: 'Itaú Unibanco',
              quantity: 800,
              avg_price: 24.00,
              current_price: 25.80,
              total_invested: 19200.00,
              current_value: 20640.00,
              return_value: 1440.00,
              return_percentage: 7.50,
              allocation: 16.5
            },
            {
              symbol: 'BBDC4',
              name: 'Bradesco',
              quantity: 1000,
              avg_price: 15.50,
              current_price: 14.25,
              total_invested: 15500.00,
              current_value: 14250.00,
              return_value: -1250.00,
              return_percentage: -8.06,
              allocation: 11.4
            },
            {
              symbol: 'ABEV3',
              name: 'Ambev',
              quantity: 1500,
              avg_price: 10.80,
              current_price: 11.45,
              total_invested: 16200.00,
              current_value: 17175.00,
              return_value: 975.00,
              return_percentage: 6.02,
              allocation: 13.7
            },
            {
              symbol: 'WEGE3',
              name: 'WEG',
              quantity: 300,
              avg_price: 42.00,
              current_price: 45.30,
              total_invested: 12600.00,
              current_value: 13590.00,
              return_value: 990.00,
              return_percentage: 7.86,
              allocation: 10.8
            }
          ],
          allocation_by_sector: [
            { sector: 'Financeiro', percentage: 27.9, value: 34890.00 },
            { sector: 'Petróleo e Gás', percentage: 12.9, value: 16225.00 },
            { sector: 'Mineração', percentage: 11.0, value: 13780.00 },
            { sector: 'Bebidas', percentage: 13.7, value: 17175.00 },
            { sector: 'Industrial', percentage: 10.8, value: 13590.00 },
            { sector: 'Outros', percentage: 23.7, value: 29770.00 }
          ]
        })
      }
    } catch (error) {
      console.error('Erro ao carregar portfólio:', error)
      // Dados simulados em caso de erro
      setPortfolio({
        total_value: 125430.00,
        total_invested: 110000.00,
        total_return: 15430.00,
        return_percentage: 14.03,
        assets: [],
        allocation_by_sector: []
      })
    } finally {
      setLoading(false)
    }
  }

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value)
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Portfólio</h1>
        <p className="text-muted-foreground">
          Acompanhe o desempenho dos seus investimentos
        </p>
      </div>

      {/* Resumo do Portfólio */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Valor Total</CardTitle>
            <Briefcase className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(portfolio?.total_value || 0)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Valor Investido</CardTitle>
            <TrendingUp className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">
              {formatCurrency(portfolio?.total_invested || 0)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Retorno Total</CardTitle>
            {(portfolio?.total_return || 0) >= 0 ? (
              <TrendingUp className="h-4 w-4 text-green-600" />
            ) : (
              <TrendingDown className="h-4 w-4 text-red-600" />
            )}
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${
              (portfolio?.total_return || 0) >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {formatCurrency(portfolio?.total_return || 0)}
            </div>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Rentabilidade</CardTitle>
            <PieChart className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className={`text-2xl font-bold ${
              (portfolio?.return_percentage || 0) >= 0 ? 'text-green-600' : 'text-red-600'
            }`}>
              {(portfolio?.return_percentage || 0) >= 0 ? '+' : ''}{(portfolio?.return_percentage || 0).toFixed(2)}%
            </div>
          </CardContent>
        </Card>
      </div>

      <div className="grid gap-6 md:grid-cols-2">
        {/* Ativos */}
        <Card>
          <CardHeader>
            <CardTitle>Ativos em Carteira</CardTitle>
            <CardDescription>
              Detalhamento dos seus investimentos
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {portfolio?.assets?.map((asset) => (
                <div key={asset.symbol} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="w-10 h-10 bg-blue-100 dark:bg-blue-900 rounded-full flex items-center justify-center">
                      <span className="text-sm font-bold text-blue-600 dark:text-blue-400">
                        {asset.symbol.slice(0, 2)}
                      </span>
                    </div>
                    <div>
                      <p className="font-medium">{asset.symbol}</p>
                      <p className="text-sm text-muted-foreground">{asset.name}</p>
                      <p className="text-xs text-muted-foreground">
                        {asset.quantity} ações • PM: {formatCurrency(asset.avg_price)}
                      </p>
                    </div>
                  </div>
                  <div className="text-right">
                    <p className="font-medium">{formatCurrency(asset.current_value)}</p>
                    <div className={`text-sm ${
                      asset.return_percentage >= 0 ? 'text-green-600' : 'text-red-600'
                    }`}>
                      {asset.return_percentage >= 0 ? '+' : ''}{asset.return_percentage.toFixed(2)}%
                    </div>
                    <p className="text-xs text-muted-foreground">
                      {asset.allocation.toFixed(1)}% da carteira
                    </p>
                  </div>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>

        {/* Alocação por Setor */}
        <Card>
          <CardHeader>
            <CardTitle>Alocação por Setor</CardTitle>
            <CardDescription>
              Distribuição dos investimentos por setor
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {portfolio?.allocation_by_sector?.map((sector, index) => (
                <div key={sector.sector} className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="text-sm font-medium">{sector.sector}</span>
                    <div className="text-right">
                      <span className="text-sm font-medium">{sector.percentage.toFixed(1)}%</span>
                      <p className="text-xs text-muted-foreground">
                        {formatCurrency(sector.value)}
                      </p>
                    </div>
                  </div>
                  <Progress value={sector.percentage} className="h-2" />
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Ações Rápidas */}
      <Card>
        <CardHeader>
          <CardTitle>Ações Rápidas</CardTitle>
          <CardDescription>
            Gerencie seu portfólio
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <Button>
              <Plus className="w-4 h-4 mr-2" />
              Adicionar Ativo
            </Button>
            <Button variant="outline">
              <Minus className="w-4 h-4 mr-2" />
              Remover Ativo
            </Button>
            <Button variant="outline">
              <TrendingUp className="w-4 h-4 mr-2" />
              Rebalancear
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

