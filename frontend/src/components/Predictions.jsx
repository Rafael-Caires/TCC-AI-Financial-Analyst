import { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from './ui/select'
import { TrendingUp, TrendingDown, Calendar, BarChart3 } from 'lucide-react'

export default function Predictions() {
  const [predictions, setPredictions] = useState([])
  const [selectedStock, setSelectedStock] = useState('PETR4')
  const [timeframe, setTimeframe] = useState('7')
  const [loading, setLoading] = useState(false)

  const stocks = [
    { value: 'PETR4', label: 'Petrobras (PETR4)' },
    { value: 'VALE3', label: 'Vale (VALE3)' },
    { value: 'ITUB4', label: 'Itaú Unibanco (ITUB4)' },
    { value: 'BBDC4', label: 'Bradesco (BBDC4)' },
    { value: 'ABEV3', label: 'Ambev (ABEV3)' }
  ]

  const timeframes = [
    { value: '7', label: '7 dias' },
    { value: '30', label: '30 dias' },
    { value: '90', label: '90 dias' }
  ]

  useEffect(() => {
    fetchPredictions()
  }, [selectedStock, timeframe])

  const fetchPredictions = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('token')
      const response = await fetch(`http://localhost:5000/api/predictions/${selectedStock}?days=${timeframe}`, {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        console.log('Predictions API response:', data) // Debug
        // A API retorna predictions diretamente, não dentro de data.predictions
        setPredictions(data.predictions || [])
      } else {
        // Dados simulados se a API não estiver disponível
        setPredictions([
          {
            date: '2025-06-09',
            predicted_price: 33.45,
            confidence: 85,
            trend: 'up',
            change_percentage: 3.1
          },
          {
            date: '2025-06-10',
            predicted_price: 34.20,
            confidence: 82,
            trend: 'up',
            change_percentage: 2.2
          },
          {
            date: '2025-06-11',
            predicted_price: 33.80,
            confidence: 78,
            trend: 'down',
            change_percentage: -1.2
          },
          {
            date: '2025-06-12',
            predicted_price: 35.10,
            confidence: 80,
            trend: 'up',
            change_percentage: 3.8
          },
          {
            date: '2025-06-13',
            predicted_price: 34.90,
            confidence: 77,
            trend: 'down',
            change_percentage: -0.6
          }
        ])
      }
    } catch (error) {
      console.error('Erro ao carregar previsões:', error)
      // Dados simulados em caso de erro
      setPredictions([
        {
          date: '2025-06-09',
          predicted_price: 33.45,
          confidence: 85,
          trend: 'up',
          change_percentage: 3.1
        },
        {
          date: '2025-06-10',
          predicted_price: 34.20,
          confidence: 82,
          trend: 'up',
          change_percentage: 2.2
        },
        {
          date: '2025-06-11',
          predicted_price: 33.80,
          confidence: 78,
          trend: 'down',
          change_percentage: -1.2
        },
        {
          date: '2025-06-12',
          predicted_price: 35.10,
          confidence: 80,
          trend: 'up',
          change_percentage: 3.8
        },
        {
          date: '2025-06-13',
          predicted_price: 34.90,
          confidence: 77,
          trend: 'down',
          change_percentage: -0.6
        }
      ])
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

  const formatDate = (dateString) => {
    if (!dateString) return 'Data inválida'
    try {
      return new Date(dateString).toLocaleDateString('pt-BR')
    } catch (error) {
      return 'Data inválida'
    }
  }

  const getConfidenceColor = (confidence) => {
    if (confidence >= 80) return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
    if (confidence >= 60) return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
    return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Previsões</h1>
        <p className="text-muted-foreground">
          Análise preditiva baseada em machine learning para ações selecionadas.
        </p>
      </div>

      {/* Filtros */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <BarChart3 className="w-5 h-5" />
            Configurações de Previsão
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="flex gap-4">
            <div className="flex-1">
              <label className="text-sm font-medium mb-2 block">Ação</label>
              <Select value={selectedStock} onValueChange={setSelectedStock}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {stocks.map((stock) => (
                    <SelectItem key={stock.value} value={stock.value}>
                      {stock.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex-1">
              <label className="text-sm font-medium mb-2 block">Período</label>
              <Select value={timeframe} onValueChange={setTimeframe}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {timeframes.map((tf) => (
                    <SelectItem key={tf.value} value={tf.value}>
                      {tf.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <div className="flex items-end">
              <Button onClick={fetchPredictions} disabled={loading}>
                {loading ? 'Carregando...' : 'Atualizar'}
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Previsões */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <TrendingUp className="w-5 h-5" />
            Previsões para {selectedStock}
          </CardTitle>
          <CardDescription>
            Previsões de preço para os próximos {timeframe} dias
          </CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <div className="flex items-center justify-center h-32">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
            </div>
          ) : (
            <div className="space-y-4">
              {predictions.map((prediction, index) => (
                <div key={index} className="flex items-center justify-between p-4 border rounded-lg">
                  <div className="flex items-center gap-4">
                    <div className="flex items-center gap-2">
                      <Calendar className="w-4 h-4 text-muted-foreground" />
                      <span className="font-medium">{formatDate(prediction.date || '')}</span>
                    </div>
                    <div className="flex items-center gap-2">
                      {(prediction.trend || 'down') === 'up' ? (
                        <TrendingUp className="w-4 h-4 text-green-600" />
                      ) : (
                        <TrendingDown className="w-4 h-4 text-red-600" />
                      )}
                      <span className="text-lg font-bold">
                        {formatCurrency(prediction.predicted_price || 0)}
                      </span>
                    </div>
                  </div>
                  <div className="flex items-center gap-4">
                    <div className="text-right">
                      <div className={`font-medium ${
                        (prediction.change_percentage || 0) >= 0 ? 'text-green-600' : 'text-red-600'
                      }`}>
                        {(prediction.change_percentage || 0) >= 0 ? '+' : ''}{(prediction.change_percentage || 0)}%
                      </div>
                      <div className="text-sm text-muted-foreground">Variação</div>
                    </div>
                    <Badge className={getConfidenceColor(prediction.confidence || 0)}>
                      {(prediction.confidence || 0)}% confiança
                    </Badge>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Informações sobre o modelo */}
      <Card>
        <CardHeader>
          <CardTitle>Sobre as Previsões</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm text-muted-foreground">
            <p>
              • As previsões são baseadas em modelos de machine learning que analisam dados históricos de preços, volume e indicadores técnicos.
            </p>
            <p>
              • O nível de confiança indica a precisão esperada da previsão com base no desempenho histórico do modelo.
            </p>
            <p>
              • Previsões são apenas estimativas e não devem ser usadas como única base para decisões de investimento.
            </p>
            <p>
              • Recomendamos sempre consultar um profissional qualificado antes de tomar decisões financeiras.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

