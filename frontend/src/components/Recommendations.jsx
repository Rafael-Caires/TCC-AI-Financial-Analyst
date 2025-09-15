import React, { useState, useEffect, useMemo } from 'react'
import { useAuth } from '../contexts/AuthContext'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Input } from './ui/input'
import { Label } from './ui/label'
import { Progress } from './ui/progress'
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs'
import { Alert, AlertDescription, AlertTitle } from './ui/alert'
import { Skeleton } from './ui/skeleton'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from './ui/select'
import {
  PieChart,
  Pie,
  BarChart,
  Bar,
  LineChart,
  Line,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  CartesianGrid,
  XAxis,
  YAxis,
  Tooltip,
  Cell,
  ResponsiveContainer
} from 'recharts'
import {
  Target,
  TrendingUp,
  TrendingDown,
  AlertTriangle,
  Info,
  RefreshCw,
  Star,
  Shield,
  PieChart as PieChartIcon,
  BarChart3,
  Users,
  Brain,
  Zap,
  DollarSign,
  Percent,
  Activity,
  Filter,
  ArrowUp,
  ArrowDown,
  Minus,
  CheckCircle
} from 'lucide-react'

export default function Recommendations() {
  const { user } = useAuth()
  
  // Estados principais
  const [recommendations, setRecommendations] = useState(null)
  const [portfolioAnalysis, setPortfolioAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  // Configurações
  const [portfolioValue, setPortfolioValue] = useState('50000')
  const [userProfile, setUserProfile] = useState(user?.risk_profile || 'moderate')
  const [selectedSector, setSelectedSector] = useState('all')
  const [minScore, setMinScore] = useState(70)
  const [maxRisk, setMaxRisk] = useState('high')

  // Opções de filtros
  const sectors = [
    { value: 'all', label: 'Todos os Setores' },
    { value: 'financeiro', label: 'Financeiro' },
    { value: 'energia', label: 'Energia' },
    { value: 'mineracao', label: 'Mineração' },
    { value: 'varejo', label: 'Varejo' },
    { value: 'tecnologia', label: 'Tecnologia' },
    { value: 'saude', label: 'Saúde' },
    { value: 'industria', label: 'Industrial' }
  ]

  const riskLevels = [
    { value: 'low', label: 'Baixo' },
    { value: 'medium', label: 'Médio' },
    { value: 'high', label: 'Alto' }
  ]

  // Fetch recommendations
  const fetchRecommendations = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const response = await fetch('/api/recommendations/advanced', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({
          user_profile: userProfile,
          portfolio_value: parseFloat(portfolioValue),
          num_recommendations: 20,
          filters: {
            sector: selectedSector,
            min_score: minScore / 100,
            max_risk_level: maxRisk
          }
        })
      })

      const result = await response.json()
      
      if (result.success) {
        setRecommendations(result.data)
        
        // Carrega análise de portfólio se houver alocação sugerida
        if (result.data.suggested_allocation) {
          await fetchPortfolioAnalysis(result.data.suggested_allocation)
        }
      } else {
        setError(result.message || 'Erro ao carregar recomendações')
      }
    } catch (error) {
      console.error('Erro:', error)
      setError('Erro de conexão. Carregando dados simulados...')
      
      // Dados simulados para desenvolvimento
      setRecommendations({
        recommendations: [
          {
            ticker: 'ITUB4',
            name: 'Itaú Unibanco PN',
            sector: 'Bancos',
            score: 0.88,
            expected_return: 0.12,
            risk_level: 'medium',
            current_price: 25.80,
            target_price: 29.50,
            reasoning: 'Banco com fundamentos sólidos, crescimento consistente de receitas e posição de liderança no mercado brasileiro. Indicadores técnicos favoráveis.',
            recommendation_type: 'COMPRA',
            confidence: 0.85,
            technical_score: 0.82,
            fundamental_score: 0.90,
            sentiment_score: 0.78,
            suggested_allocation: 0.15
          },
          {
            ticker: 'PETR4',
            name: 'Petrobras PN',
            sector: 'Energia',
            score: 0.75,
            expected_return: 0.08,
            risk_level: 'high',
            current_price: 32.15,
            target_price: 35.80,
            reasoning: 'Beneficiada por preços elevados do petróleo e política de dividendos atrativa. Risco político permanece elevado.',
            recommendation_type: 'COMPRA',
            confidence: 0.72,
            technical_score: 0.78,
            fundamental_score: 0.85,
            sentiment_score: 0.62,
            suggested_allocation: 0.10
          },
          {
            ticker: 'VALE3',
            name: 'Vale ON',
            sector: 'Mineração',
            score: 0.82,
            expected_return: 0.15,
            risk_level: 'high',
            current_price: 68.90,
            target_price: 78.50,
            reasoning: 'Maior mineradora do mundo, beneficiada por demanda chinesa por minério de ferro. Forte geração de caixa.',
            recommendation_type: 'COMPRA',
            confidence: 0.80,
            technical_score: 0.85,
            fundamental_score: 0.88,
            sentiment_score: 0.75,
            suggested_allocation: 0.12
          },
          {
            ticker: 'MGLU3',
            name: 'Magazine Luiza ON',
            sector: 'Varejo',
            score: 0.65,
            expected_return: 0.05,
            risk_level: 'high',
            current_price: 4.25,
            target_price: 6.80,
            reasoning: 'Setor de varejo em recuperação. Estratégia digital bem executada, mas alta competição no e-commerce.',
            recommendation_type: 'NEUTRO',
            confidence: 0.60,
            technical_score: 0.58,
            fundamental_score: 0.65,
            sentiment_score: 0.72,
            suggested_allocation: 0.05
          },
          {
            ticker: 'WEGE3',
            name: 'WEG ON',
            sector: 'Bens Industriais',
            score: 0.91,
            expected_return: 0.18,
            risk_level: 'medium',
            current_price: 45.20,
            target_price: 53.80,
            reasoning: 'Empresa de excelência operacional com forte presença internacional. Beneficiada por investimentos em infraestrutura.',
            recommendation_type: 'COMPRA',
            confidence: 0.92,
            technical_score: 0.89,
            fundamental_score: 0.95,
            sentiment_score: 0.88,
            suggested_allocation: 0.18
          }
        ],
        portfolio_metrics: {
          total_score: 0.82,
          expected_return: 0.14,
          diversification_score: 0.78,
          total_allocation: 0.60,
          risk_score: 0.72
        },
        suggested_allocation: {
          allocations: [
            { ticker: 'WEGE3', weight: 0.18 },
            { ticker: 'ITUB4', weight: 0.15 },
            { ticker: 'VALE3', weight: 0.12 },
            { ticker: 'PETR4', weight: 0.10 },
            { ticker: 'MGLU3', weight: 0.05 }
          ]
        }
      })
    } finally {
      setLoading(false)
    }
  }

  const fetchPortfolioAnalysis = async (allocation) => {
    try {
      const portfolioWeights = {}
      allocation.allocations.forEach(item => {
        portfolioWeights[item.ticker] = item.weight
      })

      const response = await fetch('/api/risk-analysis/portfolio', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ portfolio_weights: portfolioWeights })
      })

      const result = await response.json()
      
      if (result.success) {
        setPortfolioAnalysis(result.data)
      } else {
        // Dados simulados
        setPortfolioAnalysis({
          expected_return: 0.14,
          volatility: 0.22,
          sharpe_ratio: 0.64,
          var_95: -0.045,
          cvar_95: -0.062,
          max_drawdown: -0.18,
          beta: 1.15,
          alpha: 0.02
        })
      }
    } catch (error) {
      console.error('Erro ao carregar análise de portfólio:', error)
    }
  }

  // Dados computados
  const filteredRecommendations = useMemo(() => {
    if (!recommendations?.recommendations) return []
    
    return recommendations.recommendations.filter(rec => {
      if (selectedSector !== 'all' && rec.sector.toLowerCase() !== selectedSector) return false
      if ((rec.score * 100) < minScore) return false
      if (maxRisk === 'low' && rec.risk_level !== 'low') return false
      if (maxRisk === 'medium' && rec.risk_level === 'high') return false
      return true
    })
  }, [recommendations, selectedSector, minScore, maxRisk])

  const sectorDistribution = useMemo(() => {
    if (!filteredRecommendations.length) return []
    
    const distribution = {}
    filteredRecommendations.forEach(rec => {
      const sector = rec.sector || 'Outros'
      distribution[sector] = (distribution[sector] || 0) + (rec.suggested_allocation || 0)
    })
    
    return Object.entries(distribution).map(([sector, value]) => ({
      sector,
      value: value * 100,
      count: filteredRecommendations.filter(r => r.sector === sector).length
    }))
  }, [filteredRecommendations])

  const riskReturnData = useMemo(() => {
    if (!filteredRecommendations.length) return []
    
    return filteredRecommendations.map(rec => ({
      ticker: rec.ticker,
      risk: rec.risk_level === 'low' ? 1 : rec.risk_level === 'medium' ? 2 : 3,
      return: (rec.expected_return || 0) * 100,
      score: (rec.score || 0) * 100
    }))
  }, [filteredRecommendations])

  // Utilities
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value)
  }

  const getRecommendationIcon = (type) => {
    switch (type?.toUpperCase()) {
      case 'COMPRA':
        return <TrendingUp className="w-4 h-4 text-green-600" />
      case 'VENDA':
        return <TrendingDown className="w-4 h-4 text-red-600" />
      case 'NEUTRO':
        return <Minus className="w-4 h-4 text-yellow-600" />
      default:
        return <Target className="w-4 h-4" />
    }
  }

  const getRiskBadgeColor = (riskLevel) => {
    switch (riskLevel) {
      case 'low':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'medium':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      case 'high':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00ff00', '#ff00ff']

  useEffect(() => {
    fetchRecommendations()
  }, [])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Recomendações IA</h1>
          <p className="text-muted-foreground">
            Sistema híbrido de recomendações baseado em ML, análise técnica e fundamentalista
          </p>
        </div>
        
        <Button onClick={fetchRecommendations} disabled={loading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Atualizar
        </Button>
      </div>

      {/* Controles e Filtros */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Filter className="h-5 w-5" />
            Configurações e Filtros
          </CardTitle>
          <CardDescription>
            Configure seu perfil e filtros para obter recomendações personalizadas
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <div className="space-y-2">
              <Label htmlFor="portfolio-value">Valor do Portfólio</Label>
              <Input
                id="portfolio-value"
                type="number"
                value={portfolioValue}
                onChange={(e) => setPortfolioValue(e.target.value)}
                placeholder="Ex: 50000"
              />
            </div>
            
            <div className="space-y-2">
              <Label>Perfil de Risco</Label>
              <Select value={userProfile} onValueChange={setUserProfile}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="conservative">Conservador</SelectItem>
                  <SelectItem value="moderate">Moderado</SelectItem>
                  <SelectItem value="aggressive">Agressivo</SelectItem>
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label>Setor</Label>
              <Select value={selectedSector} onValueChange={setSelectedSector}>
                <SelectTrigger>
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  {sectors.map((sector) => (
                    <SelectItem key={sector.value} value={sector.value}>
                      {sector.label}
                    </SelectItem>
                  ))}
                </SelectContent>
              </Select>
            </div>
            
            <div className="space-y-2">
              <Label>Score Mínimo: {minScore}%</Label>
              <input
                type="range"
                min="0"
                max="100"
                value={minScore}
                onChange={(e) => setMinScore(Number(e.target.value))}
                className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
              />
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Error Alert */}
      {error && (
        <Alert>
          <AlertTriangle className="h-4 w-4" />
          <AlertTitle>Atenção</AlertTitle>
          <AlertDescription>{error}</AlertDescription>
        </Alert>
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
                <Skeleton className="h-[100px] w-full" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Content Tabs */}
      {!loading && recommendations && (
        <Tabs defaultValue="recommendations" className="w-full">
          <TabsList className="grid w-full grid-cols-3">
            <TabsTrigger value="recommendations">Recomendações</TabsTrigger>
            <TabsTrigger value="portfolio">Portfólio Sugerido</TabsTrigger>
            <TabsTrigger value="analytics">Analytics</TabsTrigger>
          </TabsList>
          
          {/* Tab 1: Recommendations */}
          <TabsContent value="recommendations" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-4 gap-4 mb-6">
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Total de Recomendações</p>
                      <p className="text-2xl font-bold">{filteredRecommendations.length}</p>
                    </div>
                    <Target className="h-8 w-8 text-blue-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Score Médio</p>
                      <p className="text-2xl font-bold">
                        {filteredRecommendations.length > 0 
                          ? (filteredRecommendations.reduce((sum, r) => sum + (r.score || 0), 0) / filteredRecommendations.length * 100).toFixed(0)
                          : 0}%
                      </p>
                    </div>
                    <Star className="h-8 w-8 text-yellow-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Retorno Esperado</p>
                      <p className="text-2xl font-bold text-green-600">
                        {filteredRecommendations.length > 0 
                          ? (filteredRecommendations.reduce((sum, r) => sum + (r.expected_return || 0), 0) / filteredRecommendations.length * 100).toFixed(1)
                          : 0}%
                      </p>
                    </div>
                    <TrendingUp className="h-8 w-8 text-green-500" />
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-sm font-medium text-muted-foreground">Diversificação</p>
                      <p className="text-2xl font-bold">{sectorDistribution.length} setores</p>
                    </div>
                    <PieChartIcon className="h-8 w-8 text-purple-500" />
                  </div>
                </CardContent>
              </Card>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
              {filteredRecommendations.map((rec, index) => (
                <Card key={rec.ticker} className="hover:shadow-lg transition-shadow">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        {getRecommendationIcon(rec.recommendation_type)}
                        <div>
                          <CardTitle className="text-lg">{rec.ticker}</CardTitle>
                          <CardDescription className="text-sm">
                            {rec.name}
                          </CardDescription>
                        </div>
                      </div>
                      <Badge variant="outline" className="text-xs">
                        {rec.sector}
                      </Badge>
                    </div>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    {/* Score with Stars */}
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Score Geral</span>
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

                    {/* Price and Target */}
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-xs text-muted-foreground">Preço Atual</p>
                        <p className="text-sm font-bold">{formatCurrency(rec.current_price)}</p>
                      </div>
                      <div>
                        <p className="text-xs text-muted-foreground">Preço Alvo</p>
                        <p className="text-sm font-bold text-green-600">
                          {formatCurrency(rec.target_price)}
                        </p>
                      </div>
                    </div>

                    {/* Expected Return */}
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Retorno Esperado</span>
                      <span className="text-sm font-medium text-green-600">
                        +{((rec.expected_return || 0) * 100).toFixed(1)}%
                      </span>
                    </div>

                    {/* Risk Level */}
                    <div className="flex items-center justify-between">
                      <span className="text-sm text-muted-foreground">Risco</span>
                      <Badge className={getRiskBadgeColor(rec.risk_level)}>
                        {rec.risk_level === 'low' ? 'Baixo' : 
                         rec.risk_level === 'medium' ? 'Médio' : 'Alto'}
                      </Badge>
                    </div>

                    {/* Confidence */}
                    <div className="space-y-1">
                      <div className="flex justify-between text-sm">
                        <span className="text-muted-foreground">Confiança</span>
                        <span>{((rec.confidence || 0) * 100).toFixed(0)}%</span>
                      </div>
                      <Progress value={(rec.confidence || 0) * 100} className="h-2" />
                    </div>

                    {/* Suggested Allocation */}
                    {rec.suggested_allocation > 0 && (
                      <div className="flex items-center justify-between">
                        <span className="text-sm text-muted-foreground">Alocação Sugerida</span>
                        <span className="text-sm font-medium">
                          {((rec.suggested_allocation || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    )}

                    {/* Reasoning */}
                    <div className="pt-2 border-t">
                      <p className="text-xs text-muted-foreground">
                        <strong>Análise:</strong> {rec.reasoning}
                      </p>
                    </div>

                    {/* Sub-scores */}
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="text-center">
                        <p className="text-muted-foreground">Técnico</p>
                        <p className="font-medium">{((rec.technical_score || 0) * 100).toFixed(0)}%</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground">Fundamento</p>
                        <p className="font-medium">{((rec.fundamental_score || 0) * 100).toFixed(0)}%</p>
                      </div>
                      <div className="text-center">
                        <p className="text-muted-foreground">Sentimento</p>
                        <p className="font-medium">{((rec.sentiment_score || 0) * 100).toFixed(0)}%</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          </TabsContent>
          
          {/* Tab 2: Portfolio */}
          <TabsContent value="portfolio" className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
              {/* Portfolio Allocation */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <PieChartIcon className="h-5 w-5" />
                    Alocação por Setor
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <ResponsiveContainer width="100%" height={300}>
                    <PieChart>
                      <Pie
                        data={sectorDistribution}
                        dataKey="value"
                        nameKey="sector"
                        cx="50%"
                        cy="50%"
                        outerRadius={100}
                        fill="#8884d8"
                        label={({ sector, value }) => `${sector}: ${value.toFixed(1)}%`}
                      >
                        {sectorDistribution.map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Pie>
                      <Tooltip />
                    </PieChart>
                  </ResponsiveContainer>
                </CardContent>
              </Card>

              {/* Portfolio Metrics */}
              {portfolioAnalysis && (
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Métricas do Portfólio
                    </CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4">
                      <div>
                        <p className="text-sm text-muted-foreground">Retorno Esperado</p>
                        <p className="text-lg font-bold text-green-600">
                          {(portfolioAnalysis.expected_return * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Volatilidade</p>
                        <p className="text-lg font-bold text-orange-600">
                          {(portfolioAnalysis.volatility * 100).toFixed(1)}%
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">Sharpe Ratio</p>
                        <p className="text-lg font-bold text-blue-600">
                          {portfolioAnalysis.sharpe_ratio.toFixed(2)}
                        </p>
                      </div>
                      <div>
                        <p className="text-sm text-muted-foreground">VaR 95%</p>
                        <p className="text-lg font-bold text-red-600">
                          {(portfolioAnalysis.var_95 * 100).toFixed(2)}%
                        </p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}
            </div>

            {/* Individual Allocations */}
            {recommendations?.suggested_allocation?.allocations && (
              <Card>
                <CardHeader>
                  <CardTitle>Alocação Individual Sugerida</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-4">
                    {recommendations.suggested_allocation.allocations.map((allocation, index) => {
                      const rec = filteredRecommendations.find(r => r.ticker === allocation.ticker)
                      const value = (allocation.weight * parseFloat(portfolioValue))
                      
                      return (
                        <div key={allocation.ticker} className="flex items-center justify-between p-4 border rounded-lg">
                          <div className="flex items-center gap-4">
                            <div className="text-center">
                              <p className="font-bold text-lg">{allocation.ticker}</p>
                              <p className="text-xs text-muted-foreground">{rec?.sector}</p>
                            </div>
                            <div>
                              <p className="text-sm font-medium">{rec?.name}</p>
                              <p className="text-xs text-muted-foreground">
                                Score: {((rec?.score || 0) * 100).toFixed(0)}%
                              </p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="font-bold">{(allocation.weight * 100).toFixed(1)}%</p>
                            <p className="text-sm text-muted-foreground">
                              {formatCurrency(value)}
                            </p>
                          </div>
                        </div>
                      )
                    })}
                  </div>
                </CardContent>
              </Card>
            )}
          </TabsContent>
          
          {/* Tab 3: Analytics */}
          <TabsContent value="analytics" className="space-y-6">
            {/* Risk vs Return Chart */}
            <Card>
              <CardHeader>
                <CardTitle>Risco vs Retorno</CardTitle>
                <CardDescription>
                  Análise de risco-retorno das recomendações
                </CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={riskReturnData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="ticker" />
                    <YAxis />
                    <Tooltip 
                      formatter={(value, name) => [
                        name === 'return' ? `${value.toFixed(1)}%` : value,
                        name === 'return' ? 'Retorno Esperado' : 'Nível de Risco'
                      ]}
                    />
                    <Bar dataKey="return" fill="#8884d8" name="return" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            {/* Summary Statistics */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground">Recomendações de COMPRA</p>
                    <p className="text-2xl font-bold text-green-600">
                      {filteredRecommendations.filter(r => r.recommendation_type === 'COMPRA').length}
                    </p>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground">Recomendações NEUTRAS</p>
                    <p className="text-2xl font-bold text-yellow-600">
                      {filteredRecommendations.filter(r => r.recommendation_type === 'NEUTRO').length}
                    </p>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground">Alto Risco</p>
                    <p className="text-2xl font-bold text-red-600">
                      {filteredRecommendations.filter(r => r.risk_level === 'high').length}
                    </p>
                  </div>
                </CardContent>
              </Card>
              
              <Card>
                <CardContent className="pt-6">
                  <div className="text-center">
                    <p className="text-sm text-muted-foreground">Baixo Risco</p>
                    <p className="text-2xl font-bold text-green-600">
                      {filteredRecommendations.filter(r => r.risk_level === 'low').length}
                    </p>
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>
        </Tabs>
      )}

      {/* Disclaimer */}
      <Card className="border-orange-200 bg-orange-50 dark:border-orange-800 dark:bg-orange-950">
        <CardHeader>
          <CardTitle className="flex items-center gap-2 text-orange-800 dark:text-orange-200">
            <Info className="w-5 h-5" />
            Aviso Importante
          </CardTitle>
        </CardHeader>
        <CardContent className="text-orange-700 dark:text-orange-300">
          <div className="space-y-2 text-sm">
            <p>
              • As recomendações são baseadas em modelos de machine learning e análises quantitativas, mas não garantem resultados futuros.
            </p>
            <p>
              • Sempre realize sua própria pesquisa e considere sua situação financeira completa antes de tomar decisões de investimento.
            </p>
            <p>
              • Investimentos envolvem riscos e você pode perder parte ou todo o capital investido.
            </p>
            <p>
              • Recomendamos consultar um assessor financeiro qualificado para decisões importantes de investimento.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

