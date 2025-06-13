import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../components/ui/card'
import { Button } from '../components/ui/button'
import { Badge } from '../components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../components/ui/tabs'
import { Progress } from '../components/ui/progress'
import { Alert, AlertDescription } from '../components/ui/alert'
import { Input } from '../components/ui/input'
import { Label } from '../components/ui/label'
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../components/ui/select'
import { 
  TrendingUp, 
  TrendingDown, 
  Brain, 
  BarChart3, 
  AlertTriangle,
  Target,
  Zap,
  Activity,
  LineChart,
  PieChart,
  DollarSign,
  Loader2
} from 'lucide-react'

export default function AIAnalysis() {
  const { user } = useAuth()
  const [loading, setLoading] = useState(false)
  const [selectedTicker, setSelectedTicker] = useState('PETR4')
  const [analysisData, setAnalysisData] = useState(null)
  const [recommendations, setRecommendations] = useState(null)
  const [userProfile, setUserProfile] = useState({
    risk_profile: 'moderado',
    investment_goals: ['crescimento'],
    time_horizon: 'longo_prazo'
  })
  const [portfolioValue, setPortfolioValue] = useState(10000)

  const availableStocks = [
    { value: 'PETR4', label: 'Petrobras (PETR4)' },
    { value: 'VALE3', label: 'Vale (VALE3)' },
    { value: 'ITUB4', label: 'Itaú Unibanco (ITUB4)' },
    { value: 'BBDC4', label: 'Bradesco (BBDC4)' },
    { value: 'ABEV3', label: 'Ambev (ABEV3)' },
    { value: 'WEGE3', label: 'WEG (WEGE3)' },
    { value: 'MGLU3', label: 'Magazine Luiza (MGLU3)' },
    { value: 'RENT3', label: 'Localiza (RENT3)' },
    { value: 'LREN3', label: 'Lojas Renner (LREN3)' },
    { value: 'SUZB3', label: 'Suzano (SUZB3)' }
  ]

  useEffect(() => {
    if (selectedTicker) {
      loadCompleteAnalysis()
    }
  }, [selectedTicker])

  const loadCompleteAnalysis = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/ai-analysis/complete-analysis', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          ticker: selectedTicker,
          days_ahead: 30
        })
      })

      const result = await response.json()
      
      if (result.success) {
        setAnalysisData(result.data)
      } else {
        console.error('Erro na análise:', result.error)
      }
    } catch (error) {
      console.error('Erro ao carregar análise:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadRecommendations = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/ai-analysis/recommendations', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({
          user_profile: userProfile,
          portfolio_value: portfolioValue
        })
      })

      const result = await response.json()
      
      if (result.success) {
        setRecommendations(result.data)
      } else {
        console.error('Erro nas recomendações:', result.error)
      }
    } catch (error) {
      console.error('Erro ao carregar recomendações:', error)
    } finally {
      setLoading(false)
    }
  }

  const getRecommendationColor = (recommendation) => {
    switch (recommendation) {
      case 'COMPRA': return 'bg-green-100 text-green-800'
      case 'VENDA': return 'bg-red-100 text-red-800'
      default: return 'bg-yellow-100 text-yellow-800'
    }
  }

  const getRiskColor = (riskLevel) => {
    switch (riskLevel) {
      case 'Baixo': return 'text-green-600'
      case 'Moderado': return 'text-yellow-600'
      case 'Alto': return 'text-orange-600'
      case 'Muito Alto': return 'text-red-600'
      default: return 'text-gray-600'
    }
  }

  const getSentimentColor = (sentiment) => {
    if (sentiment.includes('Positivo')) return 'text-green-600'
    if (sentiment.includes('Negativo')) return 'text-red-600'
    return 'text-yellow-600'
  }

  if (loading && !analysisData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <Loader2 className="h-8 w-8 animate-spin mx-auto mb-4" />
          <p>Carregando análise com IA...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <Brain className="h-6 w-6 text-blue-600" />
          <h1 className="text-2xl font-bold">Análise com IA</h1>
        </div>
        <Button onClick={loadCompleteAnalysis} disabled={loading} variant="outline">
          {loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Zap className="h-4 w-4 mr-2" />}
          Atualizar
        </Button>
      </div>

      {/* Seletor de Ativo */}
      <Card>
        <CardHeader>
          <CardTitle>Selecionar Ativo para Análise</CardTitle>
          <CardDescription>
            Escolha um ativo para análise completa com IA
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="flex items-center space-x-4">
            <div className="flex-1">
              <Label htmlFor="ticker">Ativo</Label>
              <Select value={selectedTicker} onValueChange={setSelectedTicker}>
                <SelectTrigger>
                  <SelectValue placeholder="Selecione um ativo" />
                </SelectTrigger>
                <SelectContent>
                  {availableStocks.map((stock) => (
                    <SelectItem key={stock.value} value={stock.value}>
                      {stock.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            <Button onClick={loadCompleteAnalysis} disabled={loading}>
              Analisar
            </Button>
          </div>
        </CardContent>
      </Card>

      <Tabs defaultValue="analysis" className="space-y-4">
        <TabsList>
          <TabsTrigger value="analysis">Análise Completa</TabsTrigger>
          <TabsTrigger value="predictions">Previsões</TabsTrigger>
          <TabsTrigger value="risk">Análise de Risco</TabsTrigger>
          <TabsTrigger value="recommendations">Recomendações</TabsTrigger>
        </TabsList>

        <TabsContent value="analysis" className="space-y-6">
          {analysisData && (
            <>
              {/* Resumo Executivo */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center space-x-2">
                    <Target className="h-5 w-5" />
                    <span>Resumo Executivo - {selectedTicker}</span>
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        R$ {analysisData.executive_summary.current_price.toFixed(2)}
                      </div>
                      <div className="text-sm text-muted-foreground">Preço Atual</div>
                    </div>
                    <div className="text-center">
                      <div className="text-2xl font-bold">
                        R$ {analysisData.executive_summary.predicted_price_30d.toFixed(2)}
                      </div>
                      <div className="text-sm text-muted-foreground">Previsão 30 dias</div>
                    </div>
                    <div className="text-center">
                      <div className={`text-2xl font-bold ${analysisData.executive_summary.expected_return_30d > 0 ? 'text-green-600' : 'text-red-600'}`}>
                        {analysisData.executive_summary.expected_return_30d > 0 ? '+' : ''}
                        {analysisData.executive_summary.expected_return_30d.toFixed(2)}%
                      </div>
                      <div className="text-sm text-muted-foreground">Retorno Esperado</div>
                    </div>
                    <div className="text-center">
                      <Badge className={getRecommendationColor(analysisData.executive_summary.recommendation)}>
                        {analysisData.executive_summary.recommendation}
                      </Badge>
                      <div className="text-sm text-muted-foreground mt-1">Recomendação</div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              {/* Métricas Principais */}
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Nível de Risco</CardTitle>
                    <AlertTriangle className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className={`text-2xl font-bold ${getRiskColor(analysisData.executive_summary.risk_level)}`}>
                      {analysisData.executive_summary.risk_level}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Volatilidade: {analysisData.risk_metrics.volatility.toFixed(2)}%
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Sentimento</CardTitle>
                    <Activity className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className={`text-2xl font-bold ${getSentimentColor(analysisData.executive_summary.sentiment)}`}>
                      {analysisData.executive_summary.sentiment}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Score: {analysisData.sentiment_analysis.sentiment_score.toFixed(2)}
                    </p>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">RSI</CardTitle>
                    <BarChart3 className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {analysisData.technical_indicators.rsi.toFixed(1)}
                    </div>
                    <Progress value={analysisData.technical_indicators.rsi} className="h-2 mt-2" />
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                    <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
                    <LineChart className="h-4 w-4 text-muted-foreground" />
                  </CardHeader>
                  <CardContent>
                    <div className="text-2xl font-bold">
                      {analysisData.risk_metrics.sharpe_ratio.toFixed(2)}
                    </div>
                    <p className="text-xs text-muted-foreground">
                      Retorno ajustado ao risco
                    </p>
                  </CardContent>
                </Card>
              </div>

              {/* Indicadores Técnicos */}
              <Card>
                <CardHeader>
                  <CardTitle>Indicadores Técnicos</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">SMA 20:</span>
                        <span className="font-medium">R$ {analysisData.technical_indicators.sma_20.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">SMA 50:</span>
                        <span className="font-medium">R$ {analysisData.technical_indicators.sma_50.toFixed(2)}</span>
                      </div>
                      {analysisData.technical_indicators.sma_200 && (
                        <div className="flex justify-between">
                          <span className="text-sm">SMA 200:</span>
                          <span className="font-medium">R$ {analysisData.technical_indicators.sma_200.toFixed(2)}</span>
                        </div>
                      )}
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">MACD:</span>
                        <span className="font-medium">{analysisData.technical_indicators.macd.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">MACD Signal:</span>
                        <span className="font-medium">{analysisData.technical_indicators.macd_signal.toFixed(3)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">RSI:</span>
                        <span className="font-medium">{analysisData.technical_indicators.rsi.toFixed(1)}</span>
                      </div>
                    </div>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-sm">Bollinger Superior:</span>
                        <span className="font-medium">R$ {analysisData.technical_indicators.bb_upper.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Bollinger Inferior:</span>
                        <span className="font-medium">R$ {analysisData.technical_indicators.bb_lower.toFixed(2)}</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-sm">Volatilidade:</span>
                        <span className="font-medium">{analysisData.technical_indicators.volatility.toFixed(2)}%</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </TabsContent>

        <TabsContent value="predictions" className="space-y-6">
          {analysisData && (
            <Card>
              <CardHeader>
                <CardTitle>Previsões de Preço - {selectedTicker}</CardTitle>
                <CardDescription>
                  Previsões geradas usando modelo {analysisData.predictions.model_type}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="text-center">
                      <div className="text-lg font-bold">R$ {analysisData.predictions.base_price.toFixed(2)}</div>
                      <div className="text-sm text-muted-foreground">Preço Base</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold capitalize">{analysisData.predictions.trend_detected}</div>
                      <div className="text-sm text-muted-foreground">Tendência Detectada</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold">{(analysisData.predictions.volatility * 100).toFixed(2)}%</div>
                      <div className="text-sm text-muted-foreground">Volatilidade</div>
                    </div>
                  </div>

                  <div className="space-y-2">
                    <h4 className="font-medium">Previsões dos Próximos 10 Dias</h4>
                    <div className="space-y-2">
                      {analysisData.predictions.predictions.slice(0, 10).map((prediction, index) => (
                        <div key={index} className="flex items-center justify-between p-3 border rounded">
                          <div>
                            <div className="font-medium">Dia {prediction.day}</div>
                            <div className="text-sm text-muted-foreground">{prediction.date}</div>
                          </div>
                          <div className="text-right">
                            <div className="font-bold">R$ {prediction.predicted_price.toFixed(2)}</div>
                            <div className="text-xs text-muted-foreground">
                              Confiança: {(prediction.confidence * 100).toFixed(0)}%
                            </div>
                          </div>
                          <div className="text-right text-xs text-muted-foreground">
                            <div>R$ {prediction.lower_bound.toFixed(2)} - R$ {prediction.upper_bound.toFixed(2)}</div>
                            <div>Intervalo 95%</div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>

        <TabsContent value="risk" className="space-y-6">
          {analysisData && (
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              <Card>
                <CardHeader>
                  <CardTitle>Métricas de Risco</CardTitle>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid grid-cols-2 gap-4">
                    <div>
                      <div className="text-sm text-muted-foreground">VaR 95%</div>
                      <div className="text-lg font-bold text-red-600">
                        {analysisData.risk_metrics.var_95.toFixed(2)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">CVaR 95%</div>
                      <div className="text-lg font-bold text-red-600">
                        {analysisData.risk_metrics.cvar_95.toFixed(2)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Volatilidade</div>
                      <div className="text-lg font-bold">
                        {analysisData.risk_metrics.volatility.toFixed(2)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Max Drawdown</div>
                      <div className="text-lg font-bold text-red-600">
                        {analysisData.risk_metrics.max_drawdown.toFixed(2)}%
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Sharpe Ratio</div>
                      <div className="text-lg font-bold">
                        {analysisData.risk_metrics.sharpe_ratio.toFixed(2)}
                      </div>
                    </div>
                    <div>
                      <div className="text-sm text-muted-foreground">Beta</div>
                      <div className="text-lg font-bold">
                        {analysisData.risk_metrics.beta.toFixed(2)}
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Classificação de Risco</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="text-center space-y-4">
                    <div className={`text-3xl font-bold ${getRiskColor(analysisData.risk_metrics.risk_classification)}`}>
                      {analysisData.risk_metrics.risk_classification}
                    </div>
                    <div className="space-y-2">
                      <div className="text-sm text-muted-foreground">Baseado em:</div>
                      <ul className="text-sm space-y-1">
                        <li>• Volatilidade histórica</li>
                        <li>• Value at Risk (VaR)</li>
                        <li>• Maximum Drawdown</li>
                        <li>• Beta em relação ao mercado</li>
                      </ul>
                    </div>
                  </div>
                </CardContent>
              </Card>
            </div>
          )}
        </TabsContent>

        <TabsContent value="recommendations" className="space-y-6">
          <Card>
            <CardHeader>
              <CardTitle>Configurar Perfil para Recomendações</CardTitle>
              <CardDescription>
                Configure seu perfil de investidor para receber recomendações personalizadas
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
                <div>
                  <Label htmlFor="risk-profile">Perfil de Risco</Label>
                  <Select value={userProfile.risk_profile} onValueChange={(value) => 
                    setUserProfile(prev => ({ ...prev, risk_profile: value }))
                  }>
                    <SelectTrigger>
                      <SelectValue />
                    </SelectTrigger>
                    <SelectContent>
                      <SelectItem value="conservador">Conservador</SelectItem>
                      <SelectItem value="moderado">Moderado</SelectItem>
                      <SelectItem value="agressivo">Agressivo</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div>
                  <Label htmlFor="portfolio-value">Valor do Portfólio (R$)</Label>
                  <Input
                    type="number"
                    value={portfolioValue}
                    onChange={(e) => setPortfolioValue(Number(e.target.value))}
                    placeholder="10000"
                  />
                </div>
                <div className="flex items-end">
                  <Button onClick={loadRecommendations} disabled={loading} className="w-full">
                    {loading ? <Loader2 className="h-4 w-4 mr-2 animate-spin" /> : <Target className="h-4 w-4 mr-2" />}
                    Gerar Recomendações
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>

          {recommendations && (
            <Card>
              <CardHeader>
                <CardTitle>Recomendações Personalizadas</CardTitle>
                <CardDescription>
                  Baseado no seu perfil {recommendations.user_profile} e portfólio de R$ {recommendations.portfolio_value.toLocaleString()}
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6">
                    <div className="text-center">
                      <div className="text-lg font-bold">{recommendations.recommendations.length}</div>
                      <div className="text-sm text-muted-foreground">Ativos Recomendados</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold">{(recommendations.total_suggested_allocation * 100).toFixed(1)}%</div>
                      <div className="text-sm text-muted-foreground">Alocação Total</div>
                    </div>
                    <div className="text-center">
                      <div className="text-lg font-bold">{(recommendations.diversification_score * 100).toFixed(0)}%</div>
                      <div className="text-sm text-muted-foreground">Score Diversificação</div>
                    </div>
                  </div>

                  <div className="space-y-3">
                    {recommendations.recommendations.map((rec, index) => (
                      <div key={index} className="p-4 border rounded-lg">
                        <div className="flex items-center justify-between mb-2">
                          <div>
                            <div className="font-bold">{rec.name} ({rec.ticker})</div>
                            <div className="text-sm text-muted-foreground">{rec.sector}</div>
                          </div>
                          <div className="text-right">
                            <div className="font-bold">R$ {rec.current_price.toFixed(2)}</div>
                            <Badge variant="outline">{rec.risk_level}</Badge>
                          </div>
                        </div>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <div className="text-muted-foreground">Score</div>
                            <div className="font-medium">{(rec.recommendation_score * 100).toFixed(0)}%</div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Alocação Sugerida</div>
                            <div className="font-medium">{(rec.suggested_allocation * 100).toFixed(1)}%</div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Valor Sugerido</div>
                            <div className="font-medium">R$ {rec.suggested_value.toLocaleString()}</div>
                          </div>
                          <div>
                            <div className="text-muted-foreground">Sentimento</div>
                            <div className="font-medium">{rec.sentiment}</div>
                          </div>
                        </div>
                        <div className="mt-2 text-sm text-muted-foreground">
                          <strong>Justificativa:</strong> {rec.reasoning}
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              </CardContent>
            </Card>
          )}
        </TabsContent>
      </Tabs>
    </div>
  )
}

