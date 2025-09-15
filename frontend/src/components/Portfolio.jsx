import React, { useState, useEffect, useMemo } from 'react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Progress } from './ui/progress'
import { Input } from './ui/input'
import { Label } from './ui/label'
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
  AreaChart,
  Area,
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
  Briefcase,
  TrendingUp,
  TrendingDown,
  PieChart as PieChartIcon,
  Plus,
  Minus,
  RefreshCw,
  Shield,
  Target,
  BarChart3,
  Activity,
  DollarSign,
  Percent,
  Calendar,
  AlertTriangle,
  CheckCircle,
  Info,
  Star,
  Users,
  Zap,
  Filter,
  Download,
  Upload,
  Settings
} from 'lucide-react'

export default function Portfolio() {
  // Estados principais
  const [portfolio, setPortfolio] = useState(null)
  const [portfolioAnalysis, setPortfolioAnalysis] = useState(null)
  const [performanceHistory, setPerformanceHistory] = useState(null)
  const [benchmarkComparison, setBenchmarkComparison] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

  // Estados de configuração
  const [selectedTimeframe, setSelectedTimeframe] = useState('1Y')
  const [selectedBenchmark, setSelectedBenchmark] = useState('IBOV')
  const [rebalanceTarget, setRebalanceTarget] = useState('equal')

  // Opções de configuração
  const timeframes = [
    { value: '1M', label: '1 Mês' },
    { value: '3M', label: '3 Meses' },
    { value: '6M', label: '6 Meses' },
    { value: '1Y', label: '1 Ano' },
    { value: 'YTD', label: 'Ano Atual' },
    { value: 'ALL', label: 'Tudo' }
  ]

  const benchmarks = [
    { value: 'IBOV', label: 'Ibovespa' },
    { value: 'IBRX', label: 'IBrX' },
    { value: 'SMLL', label: 'Small Caps' },
    { value: 'CDI', label: 'CDI' },
    { value: 'IPCA', label: 'IPCA+' }
  ]

  const rebalanceOptions = [
    { value: 'equal', label: 'Peso Igual' },
    { value: 'risk_parity', label: 'Paridade de Risco' },
    { value: 'market_cap', label: 'Market Cap' },
    { value: 'custom', label: 'Personalizado' }
  ]

  // Fetch portfolio data
  const fetchPortfolioData = async () => {
    setLoading(true)
    setError(null)
    
    try {
      const token = localStorage.getItem('token')
      
      // Busca dados do portfólio
      const portfolioResponse = await fetch('/api/portfolio/detailed', {
        headers: { 'Authorization': `Bearer ${token}` }
      })
      
      if (!portfolioResponse.ok) {
        throw new Error('Erro ao carregar portfólio')
      }
      
      const portfolioData = await portfolioResponse.json()
      setPortfolio(portfolioData.data)

      // Busca análise de risco do portfólio
      await fetchPortfolioAnalysis(portfolioData.data)
      
      // Busca histórico de performance
      await fetchPerformanceHistory()
      
      // Busca comparação com benchmark
      await fetchBenchmarkComparison()
      
    } catch (error) {
      console.error('Erro:', error)
      setError('Erro de conexão. Carregando dados simulados...')
      
      // Dados simulados para desenvolvimento
      setPortfolio({
        summary: {
          total_value: 145230.50,
          total_invested: 120000.00,
          total_return: 25230.50,
          return_percentage: 21.03,
          daily_change: 1250.30,
          daily_change_percentage: 0.87
        },
        assets: [
          {
            ticker: 'PETR4',
            name: 'Petrobras PN',
            sector: 'Energia',
            quantity: 500,
            avg_price: 28.50,
            current_price: 32.45,
            total_invested: 14250.00,
            current_value: 16225.00,
            return_value: 1975.00,
            return_percentage: 13.86,
            allocation_percentage: 11.2,
            daily_change: 0.85,
            daily_change_percentage: 2.69,
            risk_score: 0.75,
            recommendation: 'HOLD'
          },
          {
            ticker: 'VALE3',
            name: 'Vale ON',
            sector: 'Mineração',
            quantity: 200,
            avg_price: 65.00,
            current_price: 71.20,
            total_invested: 13000.00,
            current_value: 14240.00,
            return_value: 1240.00,
            return_percentage: 9.54,
            allocation_percentage: 9.8,
            daily_change: 1.45,
            daily_change_percentage: 2.08,
            risk_score: 0.78,
            recommendation: 'BUY'
          },
          {
            ticker: 'ITUB4',
            name: 'Itaú Unibanco PN',
            sector: 'Financeiro',
            quantity: 800,
            avg_price: 24.00,
            current_price: 27.30,
            total_invested: 19200.00,
            current_value: 21840.00,
            return_value: 2640.00,
            return_percentage: 13.75,
            allocation_percentage: 15.0,
            daily_change: 0.45,
            daily_change_percentage: 1.68,
            risk_score: 0.65,
            recommendation: 'HOLD'
          },
          {
            ticker: 'WEGE3',
            name: 'WEG ON',
            sector: 'Industrial',
            quantity: 300,
            avg_price: 42.00,
            current_price: 48.50,
            total_invested: 12600.00,
            current_value: 14550.00,
            return_value: 1950.00,
            return_percentage: 15.48,
            allocation_percentage: 10.0,
            daily_change: 0.85,
            daily_change_percentage: 1.79,
            risk_score: 0.55,
            recommendation: 'BUY'
          },
          {
            ticker: 'BBDC4',
            name: 'Bradesco PN',
            sector: 'Financeiro',
            quantity: 1000,
            avg_price: 15.50,
            current_price: 16.25,
            total_invested: 15500.00,
            current_value: 16250.00,
            return_value: 750.00,
            return_percentage: 4.84,
            allocation_percentage: 11.2,
            daily_change: 0.15,
            daily_change_percentage: 0.93,
            risk_score: 0.68,
            recommendation: 'HOLD'
          }
        ],
        allocation_by_sector: [
          { sector: 'Financeiro', percentage: 26.2, value: 38090.00, count: 2 },
          { sector: 'Energia', percentage: 11.2, value: 16225.00, count: 1 },
          { sector: 'Mineração', percentage: 9.8, value: 14240.00, count: 1 },
          { sector: 'Industrial', percentage: 10.0, value: 14550.00, count: 1 },
          { sector: 'Outros', percentage: 42.8, value: 62125.50, count: 8 }
        ],
        metrics: {
          beta: 1.15,
          alpha: 0.025,
          sharpe_ratio: 0.68,
          volatility: 0.24,
          var_95: -0.048,
          max_drawdown: -0.18,
          correlation_ibov: 0.82
        }
      })
      
      // Dados simulados de análise
      setPortfolioAnalysis({
        risk_metrics: {
          var_95: -0.048,
          cvar_95: -0.065,
          volatility: 0.24,
          beta: 1.15,
          sharpe_ratio: 0.68,
          sortino_ratio: 0.85,
          max_drawdown: -0.18,
          calmar_ratio: 1.17
        },
        diversification: {
          herfindahl_index: 0.15,
          effective_stocks: 6.67,
          sector_concentration: 0.262,
          diversification_ratio: 0.78
        },
        performance_attribution: {
          asset_selection: 0.032,
          sector_allocation: 0.015,
          interaction_effect: -0.008,
          total_excess_return: 0.039
        }
      })
      
      // Dados simulados de performance histórica
      setPerformanceHistory({
        monthly_returns: [
          { month: 'Jan', portfolio: 2.5, benchmark: 1.8 },
          { month: 'Fev', portfolio: -1.2, benchmark: -2.1 },
          { month: 'Mar', portfolio: 4.8, benchmark: 3.2 },
          { month: 'Abr', portfolio: 1.9, benchmark: 2.4 },
          { month: 'Mai', portfolio: 3.1, benchmark: 1.9 },
          { month: 'Jun', portfolio: -0.8, benchmark: -1.5 },
          { month: 'Jul', portfolio: 2.7, benchmark: 2.1 },
          { month: 'Ago', portfolio: 1.4, benchmark: 0.9 },
          { month: 'Set', portfolio: 3.6, benchmark: 2.8 },
          { month: 'Out', portfolio: 0.9, benchmark: 1.2 },
          { month: 'Nov', portfolio: 2.1, benchmark: 1.7 },
          { month: 'Dez', portfolio: 1.8, benchmark: 1.4 }
        ],
        cumulative_performance: [
          { period: '1M', portfolio: 1.8, benchmark: 1.4 },
          { period: '3M', portfolio: 5.3, benchmark: 4.1 },
          { period: '6M', portfolio: 12.7, benchmark: 9.8 },
          { period: '1Y', portfolio: 21.0, benchmark: 16.5 },
          { period: 'YTD', portfolio: 18.2, benchmark: 14.3 }
        ]
      })
      
    } finally {
      setLoading(false)
    }
  }

  const fetchPortfolioAnalysis = async (portfolioData) => {
    try {
      const weights = {}
      portfolioData.assets.forEach(asset => {
        weights[asset.ticker] = asset.allocation_percentage / 100
      })

      const response = await fetch('/api/risk-analysis/portfolio', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        },
        body: JSON.stringify({ portfolio_weights: weights })
      })

      if (response.ok) {
        const result = await response.json()
        setPortfolioAnalysis(result.data)
      }
    } catch (error) {
      console.error('Erro ao carregar análise:', error)
    }
  }

  const fetchPerformanceHistory = async () => {
    try {
      const response = await fetch(`/api/portfolio/performance?timeframe=${selectedTimeframe}`, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      })
      
      if (response.ok) {
        const result = await response.json()
        setPerformanceHistory(result.data)
      }
    } catch (error) {
      console.error('Erro ao carregar histórico:', error)
    }
  }

  const fetchBenchmarkComparison = async () => {
    try {
      const response = await fetch(`/api/portfolio/benchmark?benchmark=${selectedBenchmark}`, {
        headers: { 'Authorization': `Bearer ${localStorage.getItem('token')}` }
      })
      
      if (response.ok) {
        const result = await response.json()
        setBenchmarkComparison(result.data)
      }
    } catch (error) {
      console.error('Erro ao carregar benchmark:', error)
    }
  }

  // Dados computados
  const topPerformers = useMemo(() => {
    if (!portfolio?.assets) return []
    return [...portfolio.assets]
      .sort((a, b) => b.return_percentage - a.return_percentage)
      .slice(0, 3)
  }, [portfolio])

  const worstPerformers = useMemo(() => {
    if (!portfolio?.assets) return []
    return [...portfolio.assets]
      .sort((a, b) => a.return_percentage - b.return_percentage)
      .slice(0, 3)
  }, [portfolio])

  const riskExposure = useMemo(() => {
    if (!portfolio?.assets) return []
    
    const riskBuckets = { low: 0, medium: 0, high: 0 }
    
    portfolio.assets.forEach(asset => {
      const risk = asset.risk_score
      if (risk < 0.4) riskBuckets.low += asset.allocation_percentage
      else if (risk < 0.7) riskBuckets.medium += asset.allocation_percentage
      else riskBuckets.high += asset.allocation_percentage
    })
    
    return [
      { risk: 'Baixo', percentage: riskBuckets.low, color: '#10B981' },
      { risk: 'Médio', percentage: riskBuckets.medium, color: '#F59E0B' },
      { risk: 'Alto', percentage: riskBuckets.high, color: '#EF4444' }
    ]
  }, [portfolio])

  // Utility functions
  const formatCurrency = (value) => {
    return new Intl.NumberFormat('pt-BR', {
      style: 'currency',
      currency: 'BRL'
    }).format(value)
  }

  const formatPercentage = (value, decimals = 2) => {
    const sign = value >= 0 ? '+' : ''
    return `${sign}${value.toFixed(decimals)}%`
  }

  const getRecommendationColor = (recommendation) => {
    switch (recommendation) {
      case 'BUY':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'SELL':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      case 'HOLD':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const getRiskColor = (riskScore) => {
    if (riskScore < 0.4) return 'text-green-600'
    if (riskScore < 0.7) return 'text-yellow-600'
    return 'text-red-600'
  }

  const COLORS = ['#8884d8', '#82ca9d', '#ffc658', '#ff7300', '#00ff00', '#ff00ff']

  useEffect(() => {
    fetchPortfolioData()
  }, [selectedTimeframe, selectedBenchmark])

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row justify-between items-start lg:items-center gap-4">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">Portfólio Avançado</h1>
          <p className="text-muted-foreground">
            Dashboard completo com análise de risco, performance e diversificação
          </p>
        </div>
        
        <div className="flex gap-3">
          <Select value={selectedTimeframe} onValueChange={setSelectedTimeframe}>
            <SelectTrigger className="w-[120px]">
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
          
          <Select value={selectedBenchmark} onValueChange={setSelectedBenchmark}>
            <SelectTrigger className="w-[120px]">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {benchmarks.map((bm) => (
                <SelectItem key={bm.value} value={bm.value}>
                  {bm.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          
          <Button onClick={fetchPortfolioData} disabled={loading}>
            <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
            Atualizar
          </Button>
        </div>
      </div>

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
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
          {[...Array(8)].map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-4 w-[120px]" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-8 w-[80px] mb-2" />
                <Skeleton className="h-3 w-[60px]" />
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Content */}
      {!loading && portfolio && (
        <>
          {/* Summary Cards */}
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Valor Total</CardTitle>
                <Briefcase className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {formatCurrency(portfolio.summary.total_value)}
                </div>
                <div className={`text-sm ${
                  portfolio.summary.daily_change >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {formatPercentage(portfolio.summary.daily_change_percentage)} hoje
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Retorno Total</CardTitle>
                <TrendingUp className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className={`text-2xl font-bold ${
                  portfolio.summary.total_return >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {formatCurrency(portfolio.summary.total_return)}
                </div>
                <div className={`text-sm ${
                  portfolio.summary.return_percentage >= 0 ? 'text-green-600' : 'text-red-600'
                }`}>
                  {formatPercentage(portfolio.summary.return_percentage)}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
                <BarChart3 className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {portfolio.metrics.sharpe_ratio.toFixed(2)}
                </div>
                <div className="text-sm text-muted-foreground">
                  Retorno ajustado ao risco
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardTitle className="text-sm font-medium">Beta</CardTitle>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">
                  {portfolio.metrics.beta.toFixed(2)}
                </div>
                <div className="text-sm text-muted-foreground">
                  vs {selectedBenchmark}
                </div>
              </CardContent>
            </Card>
          </div>

          {/* Main Content Tabs */}
          <Tabs defaultValue="overview" className="w-full">
            <TabsList className="grid w-full grid-cols-4">
              <TabsTrigger value="overview">Visão Geral</TabsTrigger>
              <TabsTrigger value="assets">Ativos</TabsTrigger>
              <TabsTrigger value="risk">Análise de Risco</TabsTrigger>
              <TabsTrigger value="performance">Performance</TabsTrigger>
            </TabsList>
            
            {/* Tab 1: Overview */}
            <TabsContent value="overview" className="space-y-6">
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Allocation by Sector */}
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
                          data={portfolio.allocation_by_sector}
                          dataKey="percentage"
                          nameKey="sector"
                          cx="50%"
                          cy="50%"
                          outerRadius={100}
                          fill="#8884d8"
                          label={({ sector, percentage }) => `${sector}: ${percentage.toFixed(1)}%`}
                        >
                          {portfolio.allocation_by_sector.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>

                {/* Risk Exposure */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Shield className="h-5 w-5" />
                      Exposição ao Risco
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <ResponsiveContainer width="100%" height={300}>
                      <PieChart>
                        <Pie
                          data={riskExposure}
                          dataKey="percentage"
                          nameKey="risk"
                          cx="50%"
                          cy="50%"
                          outerRadius={100}
                          fill="#8884d8"
                          label={({ risk, percentage }) => `${risk}: ${percentage.toFixed(1)}%`}
                        >
                          {riskExposure.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={entry.color} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </CardContent>
                </Card>
              </div>

              {/* Top and Worst Performers */}
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-green-600">
                      <TrendingUp className="h-5 w-5" />
                      Melhores Performers
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {topPerformers.map((asset, index) => (
                        <div key={asset.ticker} className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-green-100 dark:bg-green-900 rounded-full flex items-center justify-center">
                              <span className="text-xs font-bold text-green-600">
                                #{index + 1}
                              </span>
                            </div>
                            <div>
                              <p className="font-medium">{asset.ticker}</p>
                              <p className="text-xs text-muted-foreground">{asset.sector}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="font-bold text-green-600">
                              +{asset.return_percentage.toFixed(2)}%
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {formatCurrency(asset.return_value)}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>

                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2 text-red-600">
                      <TrendingDown className="h-5 w-5" />
                      Piores Performers
                    </CardTitle>
                  </CardHeader>
                  <CardContent>
                    <div className="space-y-4">
                      {worstPerformers.map((asset, index) => (
                        <div key={asset.ticker} className="flex items-center justify-between">
                          <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-red-100 dark:bg-red-900 rounded-full flex items-center justify-center">
                              <span className="text-xs font-bold text-red-600">
                                #{index + 1}
                              </span>
                            </div>
                            <div>
                              <p className="font-medium">{asset.ticker}</p>
                              <p className="text-xs text-muted-foreground">{asset.sector}</p>
                            </div>
                          </div>
                          <div className="text-right">
                            <p className="font-bold text-red-600">
                              {asset.return_percentage.toFixed(2)}%
                            </p>
                            <p className="text-xs text-muted-foreground">
                              {formatCurrency(asset.return_value)}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </CardContent>
                </Card>
              </div>
            </TabsContent>
            
            {/* Tab 2: Assets */}
            <TabsContent value="assets" className="space-y-6">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
                {portfolio.assets.map((asset) => (
                  <Card key={asset.ticker} className="hover:shadow-lg transition-shadow">
                    <CardHeader className="pb-3">
                      <div className="flex items-center justify-between">
                        <div>
                          <CardTitle className="text-lg">{asset.ticker}</CardTitle>
                          <CardDescription className="text-sm">
                            {asset.name}
                          </CardDescription>
                        </div>
                        <Badge className={getRecommendationColor(asset.recommendation)}>
                          {asset.recommendation}
                        </Badge>
                      </div>
                    </CardHeader>
                    
                    <CardContent className="space-y-3">
                      {/* Current Value and Return */}
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs text-muted-foreground">Valor Atual</p>
                          <p className="text-sm font-bold">{formatCurrency(asset.current_value)}</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Retorno</p>
                          <p className={`text-sm font-bold ${
                            asset.return_percentage >= 0 ? 'text-green-600' : 'text-red-600'
                          }`}>
                            {formatPercentage(asset.return_percentage)}
                          </p>
                        </div>
                      </div>

                      {/* Price Info */}
                      <div className="grid grid-cols-2 gap-4">
                        <div>
                          <p className="text-xs text-muted-foreground">Preço Médio</p>
                          <p className="text-sm">{formatCurrency(asset.avg_price)}</p>
                        </div>
                        <div>
                          <p className="text-xs text-muted-foreground">Preço Atual</p>
                          <p className="text-sm font-medium">{formatCurrency(asset.current_price)}</p>
                        </div>
                      </div>

                      {/* Allocation and Risk */}
                      <div className="space-y-2">
                        <div className="flex justify-between text-sm">
                          <span className="text-muted-foreground">Alocação</span>
                          <span>{asset.allocation_percentage.toFixed(1)}%</span>
                        </div>
                        <Progress value={asset.allocation_percentage} className="h-2" />
                      </div>

                      <div className="flex items-center justify-between">
                        <span className="text-xs text-muted-foreground">Score de Risco</span>
                        <span className={`text-xs font-medium ${getRiskColor(asset.risk_score)}`}>
                          {(asset.risk_score * 100).toFixed(0)}/100
                        </span>
                      </div>

                      {/* Daily Change */}
                      <div className="flex items-center justify-between pt-2 border-t">
                        <span className="text-xs text-muted-foreground">Hoje</span>
                        <span className={`text-xs font-medium ${
                          asset.daily_change_percentage >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {formatPercentage(asset.daily_change_percentage)}
                        </span>
                      </div>

                      {/* Quantity and Sector */}
                      <div className="grid grid-cols-2 gap-4 text-xs text-muted-foreground">
                        <div>
                          <span>Quantidade: {asset.quantity}</span>
                        </div>
                        <div>
                          <Badge variant="outline" className="text-xs">
                            {asset.sector}
                          </Badge>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            </TabsContent>
            
            {/* Tab 3: Risk Analysis */}
            <TabsContent value="risk" className="space-y-6">
              {portfolioAnalysis && (
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                  {/* Risk Metrics */}
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
                          <p className="text-sm text-muted-foreground">VaR 95%</p>
                          <p className="text-lg font-bold text-red-600">
                            {portfolioAnalysis?.risk_metrics?.var_95 
                              ? (portfolioAnalysis.risk_metrics.var_95 * 100).toFixed(2) + '%'
                              : 'N/A'}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">CVaR 95%</p>
                          <p className="text-lg font-bold text-red-700">
                            {portfolioAnalysis?.risk_metrics?.cvar_95
                              ? (portfolioAnalysis.risk_metrics.cvar_95 * 100).toFixed(2) + '%'
                              : 'N/A'}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Volatilidade</p>
                          <p className="text-lg font-bold text-orange-600">
                            {portfolioAnalysis?.risk_metrics?.volatility
                              ? (portfolioAnalysis.risk_metrics.volatility * 100).toFixed(2) + '%'
                              : 'N/A'}
                          </p>
                        </div>
                        <div>
                          <p className="text-sm text-muted-foreground">Max Drawdown</p>
                          <p className="text-lg font-bold text-red-800">
                            {portfolioAnalysis?.risk_metrics?.max_drawdown
                              ? (portfolioAnalysis.risk_metrics.max_drawdown * 100).toFixed(2) + '%'
                              : 'N/A'}
                          </p>
                        </div>
                      </div>
                    </CardContent>
                  </Card>

                  {/* Diversification Metrics */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <Target className="h-5 w-5" />
                        Diversificação
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      <div className="space-y-3">
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Índice Herfindahl</span>
                          <span className="font-medium">
                            {portfolioAnalysis?.diversification?.herfindahl_index?.toFixed(3) || 'N/A'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Ações Efetivas</span>
                          <span className="font-medium">
                            {portfolioAnalysis?.diversification?.effective_stocks?.toFixed(1) || 'N/A'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Concentração Setorial</span>
                          <span className="font-medium">
                            {portfolioAnalysis?.diversification?.sector_concentration 
                              ? (portfolioAnalysis.diversification.sector_concentration * 100).toFixed(1) + '%'
                              : 'N/A'}
                          </span>
                        </div>
                        <div className="flex justify-between items-center">
                          <span className="text-sm text-muted-foreground">Ratio de Diversificação</span>
                          <span className="font-medium text-green-600">
                            {portfolioAnalysis.diversification.diversification_ratio.toFixed(2)}
                          </span>
                        </div>
                      </div>
                    </CardContent>
                  </Card>
                </div>
              )}

              {/* Risk Recommendations */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Info className="h-5 w-5" />
                    Recomendações de Risco
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="space-y-3">
                    <Alert>
                      <CheckCircle className="h-4 w-4" />
                      <AlertTitle>Diversificação Adequada</AlertTitle>
                      <AlertDescription>
                        Seu portfólio está bem diversificado com {portfolio.allocation_by_sector.length} setores diferentes.
                      </AlertDescription>
                    </Alert>
                    
                    {portfolio.metrics.beta > 1.2 && (
                      <Alert>
                        <AlertTriangle className="h-4 w-4" />
                        <AlertTitle>Beta Elevado</AlertTitle>
                        <AlertDescription>
                          Seu portfólio tem beta de {portfolio.metrics.beta.toFixed(2)}, indicando maior volatilidade que o mercado.
                        </AlertDescription>
                      </Alert>
                    )}
                  </div>
                </CardContent>
              </Card>
            </TabsContent>
            
            {/* Tab 4: Performance */}
            <TabsContent value="performance" className="space-y-6">
              {performanceHistory && (
                <>
                  {/* Monthly Returns Chart */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Performance Mensal vs {selectedBenchmark}</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <ResponsiveContainer width="100%" height={300}>
                        <BarChart data={performanceHistory.monthly_returns}>
                          <CartesianGrid strokeDasharray="3 3" />
                          <XAxis dataKey="month" />
                          <YAxis />
                          <Tooltip formatter={(value) => [`${value.toFixed(1)}%`, '']} />
                          <Bar dataKey="portfolio" fill="#8884d8" name="Portfólio" />
                          <Bar dataKey="benchmark" fill="#82ca9d" name={selectedBenchmark} />
                        </BarChart>
                      </ResponsiveContainer>
                    </CardContent>
                  </Card>

                  {/* Performance Summary */}
                  <Card>
                    <CardHeader>
                      <CardTitle>Resumo de Performance</CardTitle>
                    </CardHeader>
                    <CardContent>
                      <div className="grid grid-cols-1 md:grid-cols-5 gap-4">
                        {performanceHistory.cumulative_performance.map((perf) => (
                          <div key={perf.period} className="text-center">
                            <p className="text-sm text-muted-foreground">{perf.period}</p>
                            <p className="text-lg font-bold text-primary">
                              {formatPercentage(perf.portfolio)}
                            </p>
                            <p className="text-xs text-muted-foreground">
                              vs {formatPercentage(perf.benchmark)} ({selectedBenchmark})
                            </p>
                          </div>
                        ))}
                      </div>
                    </CardContent>
                  </Card>
                </>
              )}
            </TabsContent>
          </Tabs>

          {/* Action Buttons */}
          <Card>
            <CardHeader>
              <CardTitle className="flex items-center gap-2">
                <Settings className="h-5 w-5" />
                Ações do Portfólio
              </CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3">
                <Button>
                  <Plus className="w-4 h-4 mr-2" />
                  Adicionar Ativo
                </Button>
                <Button variant="outline">
                  <Target className="w-4 h-4 mr-2" />
                  Rebalancear
                </Button>
                <Button variant="outline">
                  <Download className="w-4 h-4 mr-2" />
                  Relatório PDF
                </Button>
                <Button variant="outline">
                  <Upload className="w-4 h-4 mr-2" />
                  Importar Dados
                </Button>
              </div>
            </CardContent>
          </Card>
        </>
      )}
    </div>
  )
}

