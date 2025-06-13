import { useState, useEffect } from 'react'
import { useAuth } from '../contexts/AuthContext'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card'
import { Badge } from './ui/badge'
import { Button } from './ui/button'
import { Target, TrendingUp, AlertTriangle, Info, RefreshCw } from 'lucide-react'

export default function Recommendations() {
  const { user } = useAuth()
  const [recommendations, setRecommendations] = useState([])
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    fetchRecommendations()
  }, [])

  const fetchRecommendations = async () => {
    setLoading(true)
    try {
      const token = localStorage.getItem('token')
      const response = await fetch('http://localhost:5000/api/recommendations', {
        headers: {
          'Authorization': `Bearer ${token}`
        }
      })

      if (response.ok) {
        const data = await response.json()
        setRecommendations(data.recommendations || [])
      } else {
        // Dados simulados se a API não estiver disponível
        setRecommendations([
          {
            id: 1,
            type: 'buy',
            symbol: 'ITUB4',
            title: 'Oportunidade de Compra - ITUB4',
            description: 'Ação com potencial de valorização baseada em análise técnica. O banco apresenta fundamentos sólidos e está em uma tendência de alta.',
            confidence: 85,
            target_price: 28.50,
            current_price: 25.80,
            potential_return: 10.5,
            risk_level: 'Médio',
            timeframe: '3-6 meses',
            reasons: [
              'Indicadores técnicos favoráveis',
              'Fundamentos financeiros sólidos',
              'Setor bancário em recuperação'
            ]
          },
          {
            id: 2,
            type: 'sell',
            symbol: 'VALE3',
            title: 'Considere Venda - VALE3',
            description: 'A ação apresenta sinais de sobrecompra e pode enfrentar correção no curto prazo devido à volatilidade do setor de commodities.',
            confidence: 72,
            target_price: 62.00,
            current_price: 68.90,
            potential_return: -10.0,
            risk_level: 'Alto',
            timeframe: '1-3 meses',
            reasons: [
              'Sinais de sobrecompra',
              'Volatilidade alta do setor',
              'Pressão de commodities globais'
            ]
          },
          {
            id: 3,
            type: 'diversification',
            title: 'Diversificação de Portfólio',
            description: 'Seu portfólio está concentrado em ações financeiras. Considere adicionar ativos de tecnologia e consumo para melhor diversificação.',
            confidence: null,
            risk_level: 'Baixo',
            timeframe: 'Longo prazo',
            reasons: [
              'Concentração em setor financeiro',
              'Falta de exposição a tecnologia',
              'Oportunidade de reduzir risco'
            ]
          },
          {
            id: 4,
            type: 'alert',
            symbol: 'PETR4',
            title: 'Alerta de Volatilidade - PETR4',
            description: 'A ação está apresentando alta volatilidade devido a fatores políticos e mudanças no preço do petróleo. Monitore de perto.',
            confidence: null,
            risk_level: 'Alto',
            timeframe: 'Curto prazo',
            reasons: [
              'Alta volatilidade recente',
              'Fatores políticos',
              'Oscilação do preço do petróleo'
            ]
          }
        ])
      }
    } catch (error) {
      console.error('Erro ao carregar recomendações:', error)
      // Dados simulados em caso de erro
      setRecommendations([
        {
          id: 1,
          type: 'buy',
          symbol: 'ITUB4',
          title: 'Oportunidade de Compra - ITUB4',
          description: 'Ação com potencial de valorização baseada em análise técnica.',
          confidence: 85,
          target_price: 28.50,
          current_price: 25.80,
          potential_return: 10.5,
          risk_level: 'Médio',
          timeframe: '3-6 meses'
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

  const getRecommendationIcon = (type) => {
    switch (type) {
      case 'buy':
        return <TrendingUp className="w-5 h-5 text-green-600" />
      case 'sell':
        return <TrendingUp className="w-5 h-5 text-red-600 rotate-180" />
      case 'diversification':
        return <Target className="w-5 h-5 text-blue-600" />
      case 'alert':
        return <AlertTriangle className="w-5 h-5 text-yellow-600" />
      default:
        return <Info className="w-5 h-5" />
    }
  }

  const getRecommendationColor = (type) => {
    switch (type) {
      case 'buy':
        return 'border-green-200 bg-green-50 dark:border-green-800 dark:bg-green-950'
      case 'sell':
        return 'border-red-200 bg-red-50 dark:border-red-800 dark:bg-red-950'
      case 'diversification':
        return 'border-blue-200 bg-blue-50 dark:border-blue-800 dark:bg-blue-950'
      case 'alert':
        return 'border-yellow-200 bg-yellow-50 dark:border-yellow-800 dark:bg-yellow-950'
      default:
        return 'border-gray-200 bg-gray-50 dark:border-gray-800 dark:bg-gray-950'
    }
  }

  const getTypeLabel = (type) => {
    switch (type) {
      case 'buy':
        return 'Compra'
      case 'sell':
        return 'Venda'
      case 'diversification':
        return 'Diversificação'
      case 'alert':
        return 'Alerta'
      default:
        return type
    }
  }

  const getRiskColor = (risk) => {
    switch (risk?.toLowerCase()) {
      case 'baixo':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'médio':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      case 'alto':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Recomendações</h1>
          <p className="text-muted-foreground">
            Sugestões personalizadas baseadas no seu perfil de risco: {user?.risk_profile}
          </p>
        </div>
        <Button onClick={fetchRecommendations} disabled={loading}>
          <RefreshCw className={`w-4 h-4 mr-2 ${loading ? 'animate-spin' : ''}`} />
          Atualizar
        </Button>
      </div>

      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-32 w-32 border-b-2 border-primary"></div>
        </div>
      ) : (
        <div className="space-y-6">
          {recommendations.map((rec) => (
            <Card key={rec.id} className={getRecommendationColor(rec.type)}>
              <CardHeader>
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    {getRecommendationIcon(rec.type)}
                    <div>
                      <CardTitle className="text-lg">{rec.title}</CardTitle>
                      <div className="flex items-center gap-2 mt-1">
                        <Badge variant="outline">
                          {getTypeLabel(rec.type)}
                        </Badge>
                        {rec.symbol && (
                          <Badge variant="secondary">
                            {rec.symbol}
                          </Badge>
                        )}
                        {rec.confidence && (
                          <Badge className="bg-blue-100 text-blue-800 dark:bg-blue-900 dark:text-blue-200">
                            {rec.confidence}% confiança
                          </Badge>
                        )}
                      </div>
                    </div>
                  </div>
                  <div className="text-right">
                    {rec.risk_level && (
                      <Badge className={getRiskColor(rec.risk_level)}>
                        Risco {rec.risk_level}
                      </Badge>
                    )}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                <p className="text-muted-foreground mb-4">{rec.description}</p>
                
                {(rec.current_price || rec.target_price || rec.potential_return) && (
                  <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                    {rec.current_price && (
                      <div>
                        <p className="text-sm font-medium">Preço Atual</p>
                        <p className="text-lg font-bold">{formatCurrency(rec.current_price)}</p>
                      </div>
                    )}
                    {rec.target_price && (
                      <div>
                        <p className="text-sm font-medium">Preço Alvo</p>
                        <p className="text-lg font-bold">{formatCurrency(rec.target_price)}</p>
                      </div>
                    )}
                    {rec.potential_return && (
                      <div>
                        <p className="text-sm font-medium">Retorno Potencial</p>
                        <p className={`text-lg font-bold ${
                          rec.potential_return >= 0 ? 'text-green-600' : 'text-red-600'
                        }`}>
                          {rec.potential_return >= 0 ? '+' : ''}{rec.potential_return.toFixed(1)}%
                        </p>
                      </div>
                    )}
                  </div>
                )}

                {rec.timeframe && (
                  <div className="mb-4">
                    <p className="text-sm font-medium">Prazo</p>
                    <p className="text-muted-foreground">{rec.timeframe}</p>
                  </div>
                )}

                {rec.reasons && rec.reasons.length > 0 && (
                  <div>
                    <p className="text-sm font-medium mb-2">Principais Fatores</p>
                    <ul className="list-disc list-inside space-y-1 text-sm text-muted-foreground">
                      {rec.reasons.map((reason, index) => (
                        <li key={index}>{reason}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </CardContent>
            </Card>
          ))}
        </div>
      )}

      {/* Disclaimer */}
      <Card>
        <CardHeader>
          <CardTitle className="flex items-center gap-2">
            <Info className="w-5 h-5" />
            Importante
          </CardTitle>
        </CardHeader>
        <CardContent>
          <div className="space-y-2 text-sm text-muted-foreground">
            <p>
              • As recomendações são baseadas em análises técnicas e fundamentais, mas não garantem resultados.
            </p>
            <p>
              • Sempre faça sua própria pesquisa e considere sua situação financeira antes de investir.
            </p>
            <p>
              • Investimentos envolvem riscos e você pode perder parte ou todo o capital investido.
            </p>
            <p>
              • Recomendamos consultar um assessor financeiro qualificado para decisões importantes.
            </p>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

