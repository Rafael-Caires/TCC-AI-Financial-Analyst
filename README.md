# Sistema Financeiro com Inteligência Artificial - TCC

## Autor: Rafael Lima Caires
## Data: Junho 2025
## Versão: 2.0 - Melhorias na Análise com IA

---

## 📋 Descrição do Projeto

Este projeto implementa um sistema financeiro inteligente com funcionalidades avançadas de análise com IA, conforme especificado no TCC. O sistema utiliza técnicas de machine learning para previsão de preços, análise de sentimentos, análise de risco e geração de recomendações personalizadas.

## 🚀 Principais Melhorias Implementadas

### 1. **Sistema de Análise com IA Aprimorado**
- **Previsões LSTM Simplificadas**: Implementação de modelo de previsão baseado em redes neurais LSTM
- **Análise de Sentimentos**: Sistema de análise de sentimentos de notícias e dados de mercado
- **Análise Quantitativa de Risco**: Cálculo de métricas como VaR, CVaR, Sharpe Ratio, Maximum Drawdown
- **Sistema de Recomendação Híbrido**: Recomendações personalizadas baseadas no perfil de risco do usuário

### 2. **Indicadores Técnicos Avançados**
- Médias Móveis (SMA 20, 50, 200)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Análise de volatilidade

### 3. **Interface de Usuário Melhorada**
- Dashboard interativo com análise completa
- Seletor de ativos para análise
- Visualização de previsões e métricas de risco
- Sistema de recomendações personalizadas
- Interface responsiva e moderna

### 4. **Arquitetura Robusta**
- API RESTful bem estruturada
- Tratamento de erros abrangente
- Logging detalhado para debugging
- Documentação completa dos endpoints
- Dados simulados para demonstração

## 🛠️ Tecnologias Utilizadas

### Backend
- **Python 3.11**
- **Flask** - Framework web
- **NumPy** - Computação científica
- **Pandas** - Manipulação de dados
- **Scikit-learn** - Machine learning
- **TensorFlow** - Deep learning (LSTM)
- **yFinance** - Dados financeiros
- **Flask-CORS** - Cross-origin requests

### Frontend
- **React** - Framework JavaScript
- **Tailwind CSS** - Estilização
- **Lucide Icons** - Ícones
- **Recharts** - Gráficos e visualizações

## 📦 Estrutura do Projeto

```
sistema_financeiro_ia_avancado/
├── backend/
│   ├── src/
│   │   ├── routes/
│   │   │   ├── ai_analysis.py          # 🆕 Nova API de análise com IA
│   │   │   ├── ai_routes.py            # APIs avançadas de IA
│   │   │   ├── auth.py                 # Autenticação
│   │   │   ├── stocks.py               # Dados de ações
│   │   │   ├── predictions.py          # Previsões
│   │   │   ├── recommendations.py      # Recomendações
│   │   │   └── dashboard.py            # Dashboard
│   │   ├── ml/
│   │   │   ├── ensemble_model.py       # Ensemble de modelos
│   │   │   ├── lstm_model.py           # Modelo LSTM
│   │   │   ├── sentiment_analyzer.py   # Análise de sentimentos
│   │   │   ├── risk_analyzer.py        # Análise de risco
│   │   │   └── ...
│   │   ├── services/
│   │   │   └── financial_data_service.py
│   │   ├── utils/
│   │   └── main.py                     # 🔄 Arquivo principal atualizado
│   ├── requirements.txt
│   └── README.md
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── AIAnalysis.jsx          # 🆕 Componente de análise com IA
│   │   │   └── ...
│   │   └── ...
│   ├── package.json
│   └── README.md
└── README.md                           # 🆕 Este arquivo
```

## 🔧 Instalação e Configuração

### Pré-requisitos
- Python 3.11+
- Node.js 18+
- npm ou yarn

### Backend

1. **Navegue para o diretório do backend:**
   ```bash
   cd backend
   ```

2. **Instale as dependências:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Instale bibliotecas do sistema (se necessário):**
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgomp1
   ```

4. **Execute o servidor:**
   ```bash
   python src/main.py
   ```

   O servidor estará disponível em: `http://localhost:5000`

### Frontend

1. **Navegue para o diretório do frontend:**
   ```bash
   cd frontend
   ```

2. **Instale as dependências:**
   ```bash
   npm install
   ```

3. **Execute o servidor de desenvolvimento:**
   ```bash
   npm start
   ```

   A aplicação estará disponível em: `http://localhost:3000`

## 📊 Funcionalidades da Análise com IA

### 1. **Previsão de Preços**
- **Endpoint**: `POST /api/ai-analysis/predict`
- **Funcionalidade**: Previsão de preços usando modelo LSTM simplificado
- **Parâmetros**: ticker, days_ahead (1-90 dias)

### 2. **Análise de Sentimentos**
- **Endpoint**: `POST /api/ai-analysis/sentiment`
- **Funcionalidade**: Análise de sentimentos baseada em dados de mercado
- **Parâmetros**: ticker

### 3. **Análise de Risco**
- **Endpoint**: `POST /api/ai-analysis/risk`
- **Funcionalidade**: Cálculo de métricas de risco (VaR, CVaR, Sharpe Ratio, etc.)
- **Parâmetros**: ticker

### 4. **Recomendações Personalizadas**
- **Endpoint**: `POST /api/ai-analysis/recommendations`
- **Funcionalidade**: Recomendações baseadas no perfil de risco do usuário
- **Parâmetros**: user_profile, portfolio_value

### 5. **Análise Completa**
- **Endpoint**: `POST /api/ai-analysis/complete-analysis`
- **Funcionalidade**: Análise completa combinando todas as funcionalidades
- **Parâmetros**: ticker, days_ahead

## 🎯 Ativos Disponíveis para Análise

O sistema suporta análise dos seguintes ativos brasileiros:

- **PETR4** - Petrobras
- **VALE3** - Vale
- **ITUB4** - Itaú Unibanco
- **BBDC4** - Bradesco
- **ABEV3** - Ambev
- **WEGE3** - WEG
- **MGLU3** - Magazine Luiza
- **RENT3** - Localiza
- **LREN3** - Lojas Renner
- **SUZB3** - Suzano

## 📈 Exemplo de Uso da API

### Análise Completa de um Ativo

```bash
curl -X POST http://localhost:5000/api/ai-analysis/complete-analysis \
  -H "Content-Type: application/json" \
  -d '{
    "ticker": "PETR4",
    "days_ahead": 30
  }'
```

### Recomendações Personalizadas

```bash
curl -X POST http://localhost:5000/api/ai-analysis/recommendations \
  -H "Content-Type: application/json" \
  -d '{
    "user_profile": {
      "risk_profile": "moderado",
      "investment_goals": ["crescimento"],
      "time_horizon": "longo_prazo"
    },
    "portfolio_value": 10000
  }'
```

## 🔍 Verificação de Saúde da API

```bash
curl http://localhost:5000/api/ai-analysis/health
```

## 📝 Notas Importantes

### Dados Simulados
- **Para demonstração**: O sistema utiliza dados simulados para garantir funcionamento independente de conectividade externa
- **Metodologia**: Os dados são gerados usando técnicas estatísticas que simulam comportamento real do mercado
- **Reprodutibilidade**: Utiliza seeds fixas para garantir resultados consistentes

### Perfis de Risco Suportados
- **Conservador**: Máx. 15% volatilidade, foco em dividendos
- **Moderado**: Máx. 25% volatilidade, equilíbrio risco-retorno
- **Agressivo**: Máx. 40% volatilidade, foco em crescimento

## 🚨 Troubleshooting

### Erro: "libgomp.so.1: cannot open shared object file"
```bash
sudo apt-get install -y libgomp1
```

### Erro: "No module named 'optuna'"
```bash
pip install optuna
```

### Problemas de Conectividade
- O sistema foi projetado para funcionar com dados simulados
- Não requer conectividade externa para demonstração
- Para dados reais, configure adequadamente o yFinance

## 📚 Documentação Adicional

### Arquitetura do Sistema
- **Padrão MVC**: Separação clara entre modelo, visão e controle
- **API RESTful**: Endpoints bem definidos e documentados
- **Microserviços**: Cada funcionalidade em módulo separado

### Algoritmos Implementados
- **LSTM**: Redes neurais para previsão de séries temporais
- **Random Forest**: Ensemble para previsões robustas
- **Análise Técnica**: Indicadores clássicos do mercado financeiro
- **Análise de Risco**: Métricas quantitativas modernas

## 🤝 Contribuições

Este projeto foi desenvolvido como parte do TCC e demonstra a aplicação prática de técnicas de IA no mercado financeiro.

### Principais Contribuições:
1. **Sistema híbrido** combinando múltiplas técnicas de IA
2. **Interface intuitiva** para análise financeira
3. **Arquitetura escalável** e bem documentada
4. **Implementação prática** de conceitos teóricos

## 📄 Licença

Este projeto é parte de um trabalho acadêmico (TCC) e está disponível para fins educacionais.

---

**Desenvolvido por Rafael Lima Caires - Junho 2025**

*Sistema Financeiro com Inteligência Artificial - Versão 2.0*

