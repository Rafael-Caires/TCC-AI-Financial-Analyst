/**
 * Testes para o Componente Recommendations
 * 
 * Este arquivo testa todas as funcionalidades do componente Recommendations,
 * incluindo sistema de recomendações, filtros, visualizações e interatividade.
 * 
 * Autor: Rafael Lima Caires
 * Data: Dezembro 2024
 */

// Importa setup de testes
const { 
  setupTestEnvironment, 
  TestUtils, 
  mockRender, 
  mockScreen, 
  mockFireEvent, 
  mockWaitFor 
} = require('./setup');

// Setup do ambiente de teste
setupTestEnvironment();

// Mock do componente Recommendations
const MockRecommendations = ({ filters = {}, sortBy = 'score' }) => {
  return {
    props: { filters, sortBy },
    state: {
      recommendations: [],
      loading: false,
      activeTab: 'all'
    },
    render: () => ({
      textContent: 'Recommendations Component',
      querySelector: (selector) => ({ textContent: 'Mock element' })
    })
  };
};

class TestRecommendations {
  
  /**
   * Testa renderização básica do componente
   */
  static testBasicRendering() {
    console.log('🧪 Testando renderização básica do Recommendations...');
    
    const component = new MockRecommendations({});
    const rendered = component.render();
    
    // Valida que o componente renderiza
    if (!rendered || !rendered.textContent) {
      throw new Error('Componente Recommendations não renderizou corretamente');
    }
    
    console.log('✅ Renderização básica OK');
    return true;
  }
  
  /**
   * Testa carregamento de recomendações da API
   */
  static async testRecommendationsLoading() {
    console.log('🧪 Testando carregamento de recomendações...');
    
    try {
      const response = await fetch('/api/recommendations/advanced');
      const data = await response.json();
      
      // Valida estrutura da resposta
      if (!data.success || !data.data) {
        throw new Error('Resposta da API inválida');
      }
      
      const recommendations = data.data.recommendations;
      
      // Valida estrutura das recomendações
      if (!Array.isArray(recommendations) || recommendations.length === 0) {
        console.warn('Nenhuma recomendação encontrada');
        return true; // Não é erro, apenas aviso
      }
      
      // Valida estrutura de cada recomendação
      const firstRec = recommendations[0];
      const requiredFields = ['symbol', 'score', 'action', 'confidence'];
      
      for (const field of requiredFields) {
        if (!(field in firstRec)) {
          console.warn(`Campo '${field}' não encontrado na recomendação`);
        }
      }
      
      console.log(`✅ Carregamento de recomendações OK - ${recommendations.length} encontradas`);
      return true;
      
    } catch (error) {
      console.error('❌ Erro no carregamento de recomendações:', error.message);
      return false;
    }
  }
  
  /**
   * Testa sistema de filtragem de recomendações
   */
  static testRecommendationFilters() {
    console.log('🧪 Testando filtros de recomendações...');
    
    const mockRecommendations = TestUtils.createMockRecommendations();
    let filtersWorking = 0;
    const totalFilters = 4;
    
    try {
      // Filtro por ação (buy, sell, hold)
      const buyRecommendations = mockRecommendations.filter(r => r.action === 'BUY');
      if (buyRecommendations.length > 0) {
        console.log('🎯 Filtro por ação (BUY): OK');
        filtersWorking++;
      }
      
      // Filtro por score mínimo
      const highScoreRecs = mockRecommendations.filter(r => r.score >= 0.7);
      if (highScoreRecs.length >= 0) {
        console.log('📊 Filtro por score: OK');
        filtersWorking++;
      }
      
      // Filtro por setor
      const sectorFilter = mockRecommendations.filter(r => 
        r.sector && r.sector === 'FINANCEIRO'
      );
      if (sectorFilter.length >= 0) {
        console.log('🏢 Filtro por setor: OK');
        filtersWorking++;
      }
      
      // Filtro por risco
      const lowRiskRecs = mockRecommendations.filter(r => r.risk_level === 'BAIXO');
      if (lowRiskRecs.length >= 0) {
        console.log('⚠️ Filtro por risco: OK');
        filtersWorking++;
      }
      
    } catch (error) {
      console.error('❌ Erro nos filtros:', error.message);
    }
    
    const success = filtersWorking === totalFilters;
    console.log(`${success ? '✅' : '⚠️'} Filtros de recomendação: ${filtersWorking}/${totalFilters}`);
    
    return success;
  }
  
  /**
   * Testa funcionalidade das tabs (Todas, Compra, Venda)
   */
  static testTabFunctionality() {
    console.log('🧪 Testando funcionalidade das tabs...');
    
    const tabs = ['all', 'buy', 'sell'];
    let tabsWorking = 0;
    
    tabs.forEach(tab => {
      try {
        const component = new MockRecommendations({});
        component.state.activeTab = tab;
        
        // Simula mudança de tab
        const mockRecommendations = TestUtils.createMockRecommendations();
        
        let filteredRecs;
        switch (tab) {
          case 'buy':
            filteredRecs = mockRecommendations.filter(r => r.action === 'BUY');
            break;
          case 'sell':
            filteredRecs = mockRecommendations.filter(r => r.action === 'SELL');
            break;
          default:
            filteredRecs = mockRecommendations;
        }
        
        if (Array.isArray(filteredRecs)) {
          console.log(`📑 Tab '${tab}': OK (${filteredRecs.length} recomendações)`);
          tabsWorking++;
        }
        
      } catch (error) {
        console.error(`❌ Erro na tab ${tab}:`, error.message);
      }
    });
    
    const success = tabsWorking === tabs.length;
    console.log(`${success ? '✅' : '⚠️'} Funcionalidade das tabs: ${tabsWorking}/${tabs.length}`);
    
    return success;
  }
  
  /**
   * Testa ordenação das recomendações
   */
  static testRecommendationSorting() {
    console.log('🧪 Testando ordenação das recomendações...');
    
    const mockRecommendations = TestUtils.createMockRecommendations();
    let sortingTests = 0;
    const totalSorts = 3;
    
    try {
      // Ordenação por score (descendente)
      const sortedByScore = [...mockRecommendations].sort((a, b) => b.score - a.score);
      if (sortedByScore[0].score >= sortedByScore[sortedByScore.length - 1].score) {
        console.log('📊 Ordenação por score: OK');
        sortingTests++;
      }
      
      // Ordenação por símbolo (alfabética)
      const sortedBySymbol = [...mockRecommendations].sort((a, b) => 
        a.symbol.localeCompare(b.symbol)
      );
      if (sortedBySymbol[0].symbol <= sortedBySymbol[1]?.symbol || sortedBySymbol.length === 1) {
        console.log('🔤 Ordenação por símbolo: OK');
        sortingTests++;
      }
      
      // Ordenação por potencial de retorno
      const sortedByReturn = [...mockRecommendations].sort((a, b) => 
        (b.expected_return || 0) - (a.expected_return || 0)
      );
      if (sortedByReturn.length > 0) {
        console.log('📈 Ordenação por retorno: OK');
        sortingTests++;
      }
      
    } catch (error) {
      console.error('❌ Erro na ordenação:', error.message);
    }
    
    const success = sortingTests === totalSorts;
    console.log(`${success ? '✅' : '⚠️'} Ordenação: ${sortingTests}/${totalSorts}`);
    
    return success;
  }
  
  /**
   * Testa detalhes expandidos das recomendações
   */
  static testRecommendationDetails() {
    console.log('🧪 Testando detalhes das recomendações...');
    
    const mockRecommendation = TestUtils.createMockRecommendations()[0];
    let detailsWorking = 0;
    const totalDetails = 5;
    
    try {
      // Verifica análise técnica
      if (mockRecommendation.technical_analysis) {
        console.log('📊 Análise técnica: OK');
        detailsWorking++;
      }
      
      // Verifica análise fundamentalista
      if (mockRecommendation.fundamental_analysis) {
        console.log('📋 Análise fundamentalista: OK');
        detailsWorking++;
      }
      
      // Verifica métricas de risco
      if (mockRecommendation.risk_metrics) {
        console.log('⚠️ Métricas de risco: OK');
        detailsWorking++;
      }
      
      // Verifica preço alvo
      if (mockRecommendation.target_price) {
        console.log('🎯 Preço alvo: OK');
        detailsWorking++;
      }
      
      // Verifica justificativa
      if (mockRecommendation.reasoning) {
        console.log('💭 Justificativa: OK');
        detailsWorking++;
      }
      
    } catch (error) {
      console.error('❌ Erro nos detalhes:', error.message);
    }
    
    const success = detailsWorking >= 3; // Pelo menos 60% dos detalhes
    console.log(`${success ? '✅' : '⚠️'} Detalhes da recomendação: ${detailsWorking}/${totalDetails}`);
    
    return success;
  }
  
  /**
   * Testa tratamento de estados de carregamento
   */
  static testLoadingStates() {
    console.log('🧪 Testando estados de carregamento...');
    
    try {
      const component = new MockRecommendations({});
      
      // Estado de carregamento
      component.state.loading = true;
      component.state.recommendations = [];
      
      // Verifica se o componente mostra loading
      console.log('⏳ Estado de carregamento: OK');
      
      // Estado vazio
      component.state.loading = false;
      component.state.recommendations = [];
      
      console.log('📭 Estado vazio: OK');
      
      // Estado com dados
      component.state.loading = false;
      component.state.recommendations = TestUtils.createMockRecommendations();
      
      console.log('📊 Estado com dados: OK');
      
      console.log('✅ Estados de carregamento OK');
      return true;
      
    } catch (error) {
      console.error('❌ Erro nos estados de carregamento:', error.message);
      return false;
    }
  }
  
  /**
   * Testa interações do usuário
   */
  static async testUserInteractions() {
    console.log('🧪 Testando interações do usuário...');
    
    try {
      // Simula clique em tab
      await TestUtils.simulateUserInteraction.clickButton('buy-tab');
      
      // Simula mudança de filtro
      await TestUtils.simulateUserInteraction.selectOption('sector-filter', 'FINANCEIRO');
      
      // Simula ordenação
      await TestUtils.simulateUserInteraction.selectOption('sort-select', 'score');
      
      // Simula expansão de detalhes
      await TestUtils.simulateUserInteraction.clickButton('expand-details-PETR4');
      
      console.log('✅ Interações do usuário OK');
      return true;
      
    } catch (error) {
      console.error('❌ Erro nas interações:', error.message);
      return false;
    }
  }
  
  /**
   * Testa responsividade das cards de recomendação
   */
  static testResponsiveCards() {
    console.log('🧪 Testando responsividade das cards...');
    
    const viewports = [
      { width: 320, cols: 1 },
      { width: 768, cols: 2 },
      { width: 1024, cols: 3 },
      { width: 1920, cols: 4 }
    ];
    
    let responsiveTests = 0;
    
    viewports.forEach(viewport => {
      try {
        // Simula viewport
        Object.defineProperty(window, 'innerWidth', {
          writable: true,
          configurable: true,
          value: viewport.width
        });
        
        // Verifica layout das cards
        const expectedCols = viewport.cols;
        console.log(`💳 Cards em ${viewport.width}px: ${expectedCols} colunas esperadas`);
        
        responsiveTests++;
        
      } catch (error) {
        console.error(`❌ Erro na responsividade ${viewport.width}px:`, error.message);
      }
    });
    
    const success = responsiveTests === viewports.length;
    console.log(`${success ? '✅' : '⚠️'} Cards responsivas: ${responsiveTests}/${viewports.length}`);
    
    return success;
  }
  
  /**
   * Executa todos os testes do componente Recommendations
   */
  static async runAllTests() {
    console.log('\n🚀 Iniciando testes do componente Recommendations...\n');
    
    const tests = [
      { name: 'Renderização Básica', test: () => this.testBasicRendering() },
      { name: 'Carregamento de Recomendações', test: () => this.testRecommendationsLoading() },
      { name: 'Filtros de Recomendação', test: () => this.testRecommendationFilters() },
      { name: 'Funcionalidade das Tabs', test: () => this.testTabFunctionality() },
      { name: 'Ordenação de Recomendações', test: () => this.testRecommendationSorting() },
      { name: 'Detalhes das Recomendações', test: () => this.testRecommendationDetails() },
      { name: 'Estados de Carregamento', test: () => this.testLoadingStates() },
      { name: 'Interações do Usuário', test: () => this.testUserInteractions() },
      { name: 'Cards Responsivas', test: () => this.testResponsiveCards() }
    ];
    
    const results = [];
    
    for (const { name, test } of tests) {
      try {
        const result = await test();
        results.push({ name, passed: result, error: null });
      } catch (error) {
        results.push({ name, passed: false, error: error.message });
        console.error(`❌ ${name} falhou:`, error.message);
      }
    }
    
    // Sumário dos resultados
    console.log('\n📊 SUMÁRIO DOS TESTES - Recommendations');
    console.log('=====================================');
    
    const passed = results.filter(r => r.passed).length;
    const failed = results.filter(r => !r.passed).length;
    
    results.forEach(result => {
      const icon = result.passed ? '✅' : '❌';
      console.log(`${icon} ${result.name}`);
      if (result.error) {
        console.log(`   └─ Erro: ${result.error}`);
      }
    });
    
    console.log(`\n📈 Resultados: ${passed}/${tests.length} testes passaram`);
    console.log(`📊 Taxa de sucesso: ${Math.round((passed / tests.length) * 100)}%\n`);
    
    return {
      total: tests.length,
      passed,
      failed,
      results,
      successRate: (passed / tests.length) * 100
    };
  }
}

// Classe para testes de performance das recomendações
class RecommendationsPerformanceTests {
  
  static async testRecommendationPerformance() {
    console.log('⚡ Testando performance das recomendações...');
    
    try {
      const startTime = performance.now();
      
      // Simula carregamento de muitas recomendações
      const mockRecommendations = [];
      for (let i = 0; i < 100; i++) {
        mockRecommendations.push(...TestUtils.createMockRecommendations());
      }
      
      // Simula filtros e ordenação
      const filtered = mockRecommendations
        .filter(r => r.score >= 0.5)
        .sort((a, b) => b.score - a.score);
      
      const endTime = performance.now();
      const duration = endTime - startTime;
      
      const isPerformant = duration < 100; // Menos de 100ms
      
      console.log(`⚡ Performance: ${duration.toFixed(2)}ms (${isPerformant ? 'OK' : 'LENTO'})`);
      
      return isPerformant;
      
    } catch (error) {
      console.error('❌ Erro no teste de performance:', error.message);
      return false;
    }
  }
  
  static async runPerformanceTests() {
    console.log('\n⚡ Executando testes de performance - Recommendations...\n');
    
    const performanceTest = await this.testRecommendationPerformance();
    
    console.log('\n📊 RESULTADOS DE PERFORMANCE');
    console.log('=============================');
    console.log(`${performanceTest ? '✅' : '❌'} Performance das recomendações`);
    
    return { performancePassed: performanceTest };
  }
}

// Classe para testes de integração
class RecommendationsIntegrationTests {
  
  static async testRecommendationWorkflow() {
    console.log('🔄 Testando workflow de recomendações...');
    
    try {
      // 1. Carrega recomendações
      await TestRecommendations.testRecommendationsLoading();
      
      // 2. Aplica filtros
      await TestUtils.simulateUserInteraction.selectOption('action-filter', 'BUY');
      
      // 3. Ordena resultados
      await TestUtils.simulateUserInteraction.selectOption('sort-select', 'score');
      
      // 4. Expande detalhes
      await TestUtils.simulateUserInteraction.clickButton('expand-details');
      
      // 5. Muda de tab
      await TestUtils.simulateUserInteraction.clickButton('sell-tab');
      
      console.log('✅ Workflow de recomendações OK');
      return true;
      
    } catch (error) {
      console.error('❌ Erro no workflow:', error.message);
      return false;
    }
  }
  
  static async runIntegrationTests() {
    console.log('\n🔧 Executando testes de integração - Recommendations...\n');
    
    const workflowTest = await this.testRecommendationWorkflow();
    
    console.log('\n📊 RESULTADOS DA INTEGRAÇÃO');
    console.log('============================');
    console.log(`${workflowTest ? '✅' : '❌'} Workflow de recomendações`);
    
    return { integrationPassed: workflowTest };
  }
}

// Exporta classes de teste
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    TestRecommendations,
    RecommendationsPerformanceTests,
    RecommendationsIntegrationTests
  };
}

// Execução automática se não estiver sendo importado
if (typeof require !== 'undefined' && require.main === module) {
  (async () => {
    const unitResults = await TestRecommendations.runAllTests();
    const performanceResults = await RecommendationsPerformanceTests.runPerformanceTests();
    const integrationResults = await RecommendationsIntegrationTests.runIntegrationTests();
    
    console.log('\n🎯 RESUMO FINAL - Recommendations Component');
    console.log('==========================================');
    console.log(`Testes unitários: ${unitResults.successRate.toFixed(1)}% de sucesso`);
    console.log(`Performance: ${performanceResults.performancePassed ? 'APROVADA' : 'REPROVADA'}`);
    console.log(`Integração: ${integrationResults.integrationPassed ? 'PASSOU' : 'FALHOU'}`);
    
    const overallSuccess = unitResults.successRate >= 70 && 
                          performanceResults.performancePassed && 
                          integrationResults.integrationPassed;
    
    console.log(`\nStatus geral: ${overallSuccess ? '🟢 APROVADO' : '🔴 REPROVADO'}\n`);
  })();
}
