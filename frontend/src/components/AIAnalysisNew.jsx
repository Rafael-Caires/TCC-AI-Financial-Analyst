import React, { useState, useEffect, useMemo } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from './ui/card'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Badge } from './ui/badge'
import { Skeleton } from './ui/skeleton'
import { Alert, AlertDescription, AlertTitle } from './ui/alert'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select'
import {
  AreaChart,
  Area,
  BarChart,
  Bar,
  LineChart,
  Line,
  PieChart,
  Pie,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Cell
} from 'recharts'
import {
  TrendingUp as TrendingUpIcon,
  TrendingDown,
  BarChart3,
  LineChart as LineChartIcon,
  PieChart as PieChartIcon,
  Activity,
  Brain,
  Target,
  AlertTriangle,
  Zap,
  Loader2,
  Shield,
  Info,
  CheckCircle,
  Users,
  Star,
  Calendar,
  DollarSign,
  Percent,
  Clock
} from 'lucide-react'

const AIAnalysis = () => {
  // Estados
  const [loading, setLoading] = useState(false)
  const [analysisData, setAnalysisData] = useState(null)
  const [recommendations, setRecommendations] = useState(null)
  const [portfolioAnalysis, setPortfolioAnalysis] = useState(null)
  const [marketRegime, setMarketRegime] = useState(null)

  // Configurações
  const [selectedStock, setSelectedStock] = useState('PETR4')
  const [portfolioValue, setPortfolioValue] = useState('10000')
  const [userProfile, setUserProfile] = useState('moderate')

  // Dados mockados expandidos
  const availableStocks = [
    { symbol: 'PETR4', name: 'Petrobras PN', sector: 'Energia' },
    { symbol: 'VALE3', name: 'Vale ON', sector: 'Mineração' },
    { symbol: 'ITUB4', name: 'Itaú Unibanco PN', sector: 'Bancos' },
    { symbol: 'BBDC4', name: 'Bradesco PN', sector: 'Bancos' },
    { symbol: 'MGLU3', name: 'Magazine Luiza ON', sector: 'Varejo' },
    { symbol: 'WEGE3', name: 'WEG ON', sector: 'Bens Industriais' },
    { symbol: 'JBSS3', name: 'JBS ON', sector: 'Alimentos' },
    { symbol: 'RENT3', name: 'Localiza ON', sector: 'Serviços' },
    { symbol: 'ABEV3', name: 'Ambev ON', sector: 'Bebidas' },
    { symbol: 'B3SA3', name: 'B3 ON', sector: 'Serviços Financeiros' },
    { symbol: 'KLBN11', name: 'Klabin PN', sector: 'Papel e Celulose' },
    { symbol: 'BPAC11', name: 'BTG Pactual PN', sector: 'Bancos' },
    { symbol: 'GGBR4', name: 'Gerdau PN', sector: 'Siderurgia' },
    { symbol: 'LREN3', name: 'Lojas Renner ON', sector: 'Varejo' }
  ]

  // Carregamento dos dados
  const loadAnalysis = async () => {
    setLoading(true)
    try {
      const response = await fetch(`/api/ai-analysis/complete/${selectedStock}`)
      const result = await response.json()
      
      if (result.success) {
        setAnalysisData(result.data)
        await loadMarketRegime()
      }
    } catch (error) {
      console.error('Erro ao carregar análise:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadMarketRegime = async () => {
    try {
      const response = await fetch('/api/ai-analysis/market-regime')
      const result = await response.json()
      
      if (result.success) {
        setMarketRegime(result.data)
      }
    } catch (error) {
      console.error('Erro ao carregar regime de mercado:', error)
    }
  }

  const loadRecommendations = async () => {
    setLoading(true)
    try {
      const response = await fetch('/api/recommendations/advanced', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          user_profile: userProfile,
          portfolio_value: portfolioValue,
          num_recommendations: 12
        })
      })

      const result = await response.json()
      
      if (result.success) {
        setRecommendations(result.data)
        
        // Carrega análise de risco do portfólio sugerido
        if (result.data.suggested_allocation) {
          loadPortfolioRiskAnalysis(result.data.suggested_allocation)
        }
      }
    } catch (error) {
      console.error('Erro ao carregar recomendações:', error)
    } finally {
      setLoading(false)
    }
  }

  const loadPortfolioRiskAnalysis = async (allocation) => {
    try {
      const portfolioWeights = {}
      allocation.allocations?.forEach(item => {
        portfolioWeights[item.ticker] = item.weight
      })

      const response = await fetch('/api/risk-analysis/portfolio', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ portfolio_weights: portfolioWeights })
      })

      const result = await response.json()
      
      if (result.success) {
        setPortfolioAnalysis(result.data)
      }
    } catch (error) {
      console.error('Erro ao carregar análise de portfólio:', error)
    }
  }

  // Dados computados para gráficos
  const chartData = useMemo(() => {
    if (!analysisData) return null

    // Dados históricos para gráfico de área
    const historicalData = analysisData.historical_prices?.slice(-30).map((price, index) => ({
      day: index + 1,
      price: price,
      sma20: analysisData.technical_indicators?.sma_20,
      sma50: analysisData.technical_indicators?.sma_50
    })) || []

    // Dados de previsão para gráfico de linha
    const predictionData = analysisData.predictions?.predictions?.slice(0, 10).map(pred => ({
      date: pred.date,
      predicted: pred.predicted_price,
      lower: pred.lower_bound,
      upper: pred.upper_bound,
      confidence: pred.confidence * 100
    })) || []

    // Dados para radar de métricas
    const radarData = [{
      metric: 'Retorno',
      value: Math.min(100, Math.max(0, (analysisData.executive_summary?.expected_return_30d || 0) * 5 + 50))
    }, {
      metric: 'Risco',
      value: Math.min(100, Math.max(0, 100 - (analysisData.risk_metrics?.volatility || 20) * 2))
    }, {
      metric: 'Sentimento',
      value: Math.min(100, Math.max(0, (analysisData.sentiment_analysis?.sentiment_score || 0) * 50 + 50))
    }, {
      metric: 'Momentum',
      value: Math.min(100, Math.max(0, (analysisData.technical_indicators?.rsi || 50)))
    }, {
      metric: 'Qualidade',
      value: Math.min(100, Math.max(0, (analysisData.risk_metrics?.sharpe_ratio || 0) * 20 + 50))
    }]

    return { historicalData, predictionData, radarData }
  }, [analysisData])

  const insightMessages = useMemo(() => {
    if (!analysisData) return []
    
    return [
      {
        icon: TrendingUpIcon,
        title: "Análise Técnica",
        message: `RSI em ${analysisData.technical_indicators?.rsi?.toFixed(1)} indica ${
          analysisData.technical_indicators?.rsi > 70 ? 'sobrecompra' : 
          analysisData.technical_indicators?.rsi < 30 ? 'sobrevenda' : 'neutralidade'
        }`
      },
      {
        icon: Shield,
        title: "Risco",
        message: `VaR 95% de ${analysisData.risk_metrics?.var_95?.toFixed(2)}% sugere ${
          Math.abs(analysisData.risk_metrics?.var_95) > 0.05 ? 'alta volatilidade' : 'risco controlado'
        }`
      },
      {
        icon: Activity,
        title: "Sentimento",
        message: `Sentimento ${analysisData.executive_summary?.sentiment} com score ${
          analysisData.sentiment_analysis?.sentiment_score?.toFixed(2)
        }`
      }
    ]
  }, [analysisData])

  // Efeitos
  useEffect(() => {
    loadAnalysis()
    loadRecommendations()
  }, [])

  return (
    <div className="space-y-6">
      {/* Header com controles */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Análise IA Avançada</h1>
          <p className="text-muted-foreground">
            Insights de machine learning e análise quantitativa em tempo real
          </p>
        </div>
        
        <div className="flex flex-col sm:flex-row gap-3 w-full lg:w-auto">
          {/* Seletor de ativo */}
          <Select value={selectedStock} onValueChange={setSelectedStock}>
            <SelectTrigger className="w-full sm:w-[200px]">
              <SelectValue placeholder="Selecionar ativo" />
            </SelectTrigger>
            <SelectContent>
              {availableStocks.map((stock) => (
                <SelectItem key={stock.symbol} value={stock.symbol}>
                  <div className="flex items-center gap-2">
                    <Badge variant="outline" className="text-xs">
                      {stock.sector}
                    </Badge>
                    {stock.symbol}
                  </div>
                </SelectItem>
              ))}
            </SelectContent>
          </Select>

          {/* Controles de portfólio */}
          <div className="flex gap-2">
            <Input
              type="number"
              placeholder="Valor do portfólio"
              value={portfolioValue}
              onChange={(e) => setPortfolioValue(e.target.value)}
              className="w-full sm:w-[150px]"
            />
            <Select value={userProfile} onValueChange={setUserProfile}>
              <SelectTrigger className="w-full sm:w-[130px]">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="conservative">Conservador</SelectItem>
                <SelectItem value="moderate">Moderado</SelectItem>
                <SelectItem value="aggressive">Agressivo</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {/* Botão de análise */}
          <Button 
            onClick={loadAnalysis} 
            disabled={loading}
            className="w-full sm:w-auto"
          >
            {loading ? (
              <>
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2" />
                Analisando...
              </>
            ) : (
              <>
                <BarChart3 className="h-4 w-4 mr-2" />
                Analisar
              </>
            )}
          </Button>
        </div>
      </div>

      {/* Alertas de insights */}
      {insightMessages.length > 0 && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {insightMessages.map((insight, index) => (
            <Alert key={index} className="border-l-4 border-l-blue-500">
              <insight.icon className="h-4 w-4" />
              <AlertTitle>{insight.title}</AlertTitle>
              <AlertDescription className="text-sm">
                {insight.message}
              </AlertDescription>
            </Alert>
          ))}
        </div>
      )}

      {/* Loading State */}
      {loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {[...Array(6)].map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-4 w-[150px]" />
                <Skeleton className="h-3 w-[100px]" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-[200px] w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Conteúdo principal - Análise do ativo */}
      {!loading && analysisData && (
        <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
          {/* Resumo Executivo */}
          <Card className="xl:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Info className="h-5 w-5" />
                Resumo Executivo - {selectedStock}
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4 mb-6">
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">Recomendação</p>
                  <Badge 
                    variant={
                      analysisData.executive_summary?.recommendation === 'COMPRA' ? 'default' :
                      analysisData.executive_summary?.recommendation === 'VENDA' ? 'destructive' : 'secondary'
                    }
                    className="text-sm"
                  >
                    {analysisData.executive_summary?.recommendation || 'NEUTRO'}
                  </Badge>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">Confiança</p>
                  <div className="flex items-center gap-2">
                    <div className="flex-1 bg-secondary rounded-full h-2">
                      <div 
                        className="bg-primary h-2 rounded-full transition-all"
                        style={{ 
                          width: `${(analysisData.executive_summary?.confidence || 0) * 100}%` 
                        }}
                      />
                    </div>
                    <span className="text-sm font-medium">
                      {((analysisData.executive_summary?.confidence || 0) * 100).toFixed(0)}%
                    </span>
                  </div>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">Preço Atual</p>
                  <p className="text-lg font-bold">
                    R$ {analysisData.current_price?.toFixed(2)}
                  </p>
                </div>
                <div className="space-y-2">
                  <p className="text-sm font-medium text-muted-foreground">Preço Alvo 30d</p>
                  <p className="text-lg font-bold text-primary">
                    R$ {analysisData.executive_summary?.target_price_30d?.toFixed(2)}
                  </p>
                </div>
              </div>
              
              <div className="space-y-2">
                <p className="text-sm font-medium">Análise:</p>
                <p className="text-sm text-muted-foreground leading-relaxed">
                  {analysisData.executive_summary?.summary || 
                   "Análise baseada em múltiplos modelos de machine learning considerando indicadores técnicos, fundamentalistas e análise de sentimento."}
                </p>
              </div>
            </CardContent>
          </Card>

          {/* Radar de Métricas */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Activity className="h-5 w-5" />
                Score Multidimensional
              </CardTitle>
            </CardHeader>
            <CardContent>
              {chartData?.radarData && (
                <RadarChart width={250} height={200} data={chartData.radarData}>
                  <PolarGrid />
                  <PolarAngleAxis dataKey="metric" tick={{ fontSize: 12 }} />
                  <PolarRadiusAxis domain={[0, 100]} tick={false} />
                  <Radar
                    name="Score"
                    dataKey="value"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.3}
                    strokeWidth={2}
                  />
                </RadarChart>
              )}
            </CardContent>
          </Card>

          {/* Gráfico de Preços Históricos */}
          <Card className="xl:col-span-2">
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <TrendingUpIcon className="h-5 w-5" />
                Análise Técnica (30 dias)
              </CardTitle>
            </CardHeader>
            <CardContent>
              {chartData?.historicalData && (
                <AreaChart width={600} height={300} data={chartData.historicalData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="day" />
                  <YAxis />
                  <Tooltip
                    formatter={(value, name) => [
                      `R$ ${value?.toFixed(2)}`,
                      name === 'price' ? 'Preço' : name === 'sma20' ? 'SMA 20' : 'SMA 50'
                    ]}
                  />
                  <Area
                    type="monotone"
                    dataKey="price"
                    stroke="hsl(var(--primary))"
                    fill="hsl(var(--primary))"
                    fillOpacity={0.3}
                  />
                  <Line type="monotone" dataKey="sma20" stroke="#8884d8" strokeDasharray="3 3" />
                  <Line type="monotone" dataKey="sma50" stroke="#82ca9d" strokeDasharray="3 3" />
                </AreaChart>
              )}
            </CardContent>
          </Card>

          {/* Previsões ML */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Brain className="h-5 w-5" />
                Previsões IA
              </CardTitle>
            </CardHeader>
            <CardContent>
              {chartData?.predictionData && chartData.predictionData.length > 0 ? (
                <LineChart width={250} height={200} data={chartData.predictionData.slice(0, 5)}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" tick={{ fontSize: 10 }} />
                  <YAxis />
                  <Tooltip
                    formatter={(value, name) => [
                      `R$ ${value?.toFixed(2)}`,
                      name === 'predicted' ? 'Previsto' : name === 'lower' ? 'Limite Inf.' : 'Limite Sup.'
                    ]}
                  />
                  <Line type="monotone" dataKey="predicted" stroke="hsl(var(--primary))" strokeWidth={2} />
                  <Line type="monotone" dataKey="lower" stroke="#8884d8" strokeDasharray="2 2" />
                  <Line type="monotone" dataKey="upper" stroke="#82ca9d" strokeDasharray="2 2" />
                </LineChart>
              ) : (
                <div className="flex items-center justify-center h-[200px] text-muted-foreground">
                  <Brain className="h-8 w-8 mr-2" />
                  Carregando previsões...
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      )}

      {/* Análise de Portfólio e Recomendações */}
      {!loading && (recommendations || portfolioAnalysis) && (
        <Tabs defaultValue="recommendations" className="w-full">
          <TabsList className="grid w-full grid-cols-2">
            <TabsTrigger value="recommendations">Recomendações IA</TabsTrigger>
            <TabsTrigger value="portfolio">Análise de Portfólio</TabsTrigger>
          </TabsList>
          
          <TabsContent value="recommendations" className="space-y-6">
            {recommendations && (
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {recommendations.recommendations?.slice(0, 9).map((rec, index) => (
                  <Card key={rec.ticker || index} className="hover:shadow-md transition-shadow">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <CardTitle className="text-lg">{rec.ticker}</CardTitle>
                        <Badge variant="outline" className="text-xs">
                          {rec.sector || 'N/A'}
                        </Badge>
                      </div>
                    </CardHeader>
                    <CardContent className="space-y-3">
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Score</span>
                        <div className="flex items-center gap-2">
                          <div className="flex">
                            {[...Array(5)].map((_, i) => (
                              <Star
                                key={i}
                                className={`h-3 w-3 ${
                                  i < Math.floor((rec.score || 0) * 5) 
                                    ? 'fill-yellow-400 text-yellow-400' 
                                    : 'text-gray-300'
                                }`}
                              />
                            ))}
                          </div>
                          <span className="text-sm font-medium">
                            {((rec.score || 0) * 100).toFixed(0)}%
                          </span>
                        </div>
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Retorno Esperado</span>
                        <span className="text-sm font-medium text-green-600">
                          {((rec.expected_return || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                      
                      <div className="flex justify-between items-center">
                        <span className="text-sm text-muted-foreground">Risco</span>
                        <Badge 
                          variant={
                            (rec.risk_level || 'medium') === 'low' ? 'default' :
                            rec.risk_level === 'medium' ? 'secondary' : 'destructive'
                          }
                          className="text-xs"
                        >
                          {rec.risk_level === 'low' ? 'Baixo' : 
                           rec.risk_level === 'medium' ? 'Médio' : 'Alto'}
                        </Badge>
                      </div>
                      
                      <div className="text-xs text-muted-foreground">
                        {rec.reasoning || 'Recomendação baseada em análise quantitativa'}
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}
          </TabsContent>
          
          <TabsContent value="portfolio" className="space-y-6">
            {portfolioAnalysis && (
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Métricas de Risco */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Métricas de Risco
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">VaR 95%</p>
                        <p className="text-lg font-bold text-red-600">
                          {(portfolioAnalysis.var_95 * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">CVaR 95%</p>
                        <p className="text-lg font-bold text-red-700">
                          {(portfolioAnalysis.cvar_95 * 100).toFixed(2)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Sharpe Ratio</p>
                        <p className="text-lg font-bold text-blue-600">
                          {portfolioAnalysis.sharpe_ratio?.toFixed(2)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm font-medium text-muted-foreground">Volatilidade</p>
                        <p className="text-lg font-bold text-orange-600">
                          {(portfolioAnalysis.volatility * 100).toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>

                {/* Alocação Sugerida */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <PieChartIcon className="h-5 w-5" />
                      Alocação Sugerida
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    {recommendations?.suggested_allocation?.allocations && (
                      <PieChart width={300} height={200}>
                        <Pie
                          data={recommendations.suggested_allocation.allocations}
                          dataKey="weight"
                          nameKey="ticker"
                          cx="50%"
                          cy="50%"
                          outerRadius={80}
                          fill="hsl(var(--primary))"
                        />
                        <Tooltip
                          formatter={(value) => [`${(value * 100).toFixed(1)}%`, 'Alocação']}
                        />
                      </PieChart>
                    )}
                  </CardContent>
                </Card>
              </div>
            )}
          </TabsContent>
        </Tabs>
      )}

      {/* Estado vazio */}
      {!loading && !analysisData && (
        <Card className="text-center py-12">
          <CardContent>
            <Brain className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
            <h3 className="text-lg font-semibold mb-2">Análise IA Avançada</h3>
            <p className="text-muted-foreground mb-4">
              Selecione um ativo e clique em "Analisar" para obter insights de machine learning
            </p>
            <Button onClick={loadAnalysis} disabled={!selectedStock}>
              <BarChart3 className="h-4 w-4 mr-2" />
              Iniciar Análise
            </Button>
          </CardContent>
        </Card>
      )}
    </div>
  )
}

export default AIAnalysis
