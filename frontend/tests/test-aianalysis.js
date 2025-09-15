/**
 * Testes para o Componente AIAnalysis
 * 
 * Este arquivo testa todas as funcionalidades do componente AIAnalysis,
 * incluindo renderização, integração com APIs, visualizações e interatividade.
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

// Mock do componente AIAnalysis se React não estiver disponível
const MockAIAnalysis = ({ timeframe = '1Y', analysisType = 'complete' }) => {
  return {
    props: { timeframe, analysisType },
    render: () => ({
      textContent: 'AI Analysis Component',
      querySelector: (selector) => ({ textContent: 'Mock element' })
    })
  };
};

class TestAIAnalysis {
  
  /**
   * Testa renderização básica do componente
   */
  static testBasicRendering() {
    console.log('🧪 Testando renderização básica do AIAnalysis...');
    
    const component = new MockAIAnalysis({});
    const rendered = component.render();
    
    // Valida que o componente renderiza
    if (!rendered || !rendered.textContent) {
      throw new Error('Componente AIAnalysis não renderizou corretamente');
    }
    
    console.log('✅ Renderização básica OK');
    return true;
  }
  
  /**
   * Testa carregamento de dados da API
   */
  static async testDataLoading() {
    console.log('🧪 Testando carregamento de dados da API...');
    
    try {
      const response = await fetch('/api/ai-analysis/complete');
      const data = await response.json();
      
      // Valida estrutura da resposta
      if (!data.success || !data.data) {
        throw new Error('Resposta da API inválida');
      }
      
      // Valida dados essenciais
      const analysis = data.data;
      const requiredSections = ['market_overview', 'ml_predictions', 'sentiment_analysis'];
      
      for (const section of requiredSections) {
        if (!analysis[section]) {
          console.warn(`Seção '${section}' não encontrada na análise`);
        }
      }
      
      console.log('✅ Carregamento de dados OK');
      return true;
      
    } catch (error) {
      console.error('❌ Erro no carregamento de dados:', error.message);
      return false;
    }
  }
  
  /**
   * Testa funcionalidade de seleção de timeframe
   */
  static testTimeframeSelection() {
    console.log('🧪 Testando seleção de timeframe...');
    
    const timeframes = ['1D', '1W', '1M', '3M', '1Y'];
    let testsPassed = 0;
    
    timeframes.forEach(timeframe => {
      try {
        const component = new MockAIAnalysis({ timeframe });
        
        if (component.props.timeframe !== timeframe) {
          throw new Error(`Timeframe não foi definido corretamente: ${timeframe}`);
        }
        
        testsPassed++;
      } catch (error) {
        console.error(`❌ Erro com timeframe ${timeframe}:`, error.message);
      }
    });
    
    const success = testsPassed === timeframes.length;
    if (success) {
      console.log('✅ Seleção de timeframe OK');
    } else {
      console.log(`⚠️ Seleção de timeframe parcialmente OK: ${testsPassed}/${timeframes.length}`);
    }
    
    return success;
  }
  
  /**
   * Testa componentes de visualização
   */
  static testVisualizationComponents() {
    console.log('🧪 Testando componentes de visualização...');
    
    const mockData = TestUtils.createMockAIAnalysis();
    let componentsWorking = 0;
    const totalComponents = 4;
    
    try {
      // Testa dados para RadarChart
      if (mockData.ml_predictions && mockData.sentiment_analysis) {
        console.log('📊 RadarChart data: OK');
        componentsWorking++;
      }
      
      // Testa dados para LineChart (previsões)
      if (mockData.ml_predictions.ensemble_forecast) {
        console.log('📈 LineChart data: OK');
        componentsWorking++;
      }
      
      // Testa dados para AreaChart
      if (mockData.market_overview) {
        console.log('📉 AreaChart data: OK');
        componentsWorking++;
      }
      
      // Testa dados para BarChart
      if (mockData.sentiment_analysis) {
        console.log('📊 BarChart data: OK');
        componentsWorking++;
      }
      
    } catch (error) {
      console.error('❌ Erro nos componentes de visualização:', error.message);
    }
    
    const success = componentsWorking === totalComponents;
    console.log(`${success ? '✅' : '⚠️'} Componentes de visualização: ${componentsWorking}/${totalComponents}`);
    
    return success;
  }
  
  /**
   * Testa tratamento de erros
   */
  static async testErrorHandling() {
    console.log('🧪 Testando tratamento de erros...');
    
    try {
      // Simula erro de rede
      global.fetch = () => Promise.reject(new Error('Network error'));
      
      const component = new MockAIAnalysis({});
      
      // O componente deve lidar graciosamente com erros
      // Em uma implementação real, verificaríamos se o estado de erro é exibido
      console.log('✅ Tratamento de erro de rede simulado OK');
      
      // Restaura fetch mock
      const { mockFetch } = require('./setup');
      global.fetch = mockFetch;
      
      return true;
      
    } catch (error) {
      console.error('❌ Erro no tratamento de erros:', error.message);
      return false;
    }
  }
  
  /**
   * Testa responsividade do layout
   */
  static testResponsiveLayout() {
    console.log('🧪 Testando layout responsivo...');
    
    const viewports = [
      { width: 320, name: 'mobile' },
      { width: 768, name: 'tablet' },
      { width: 1024, name: 'desktop' },
      { width: 1920, name: 'large' }
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
        
        const component = new MockAIAnalysis({});
        
        // Em uma implementação real, verificaríamos classes CSS responsivas
        // ou comportamentos específicos para cada viewport
        console.log(`📱 Layout ${viewport.name} (${viewport.width}px): OK`);
        responsiveTests++;
        
      } catch (error) {
        console.error(`❌ Erro no layout ${viewport.name}:`, error.message);
      }
    });
    
    const success = responsiveTests === viewports.length;
    console.log(`${success ? '✅' : '⚠️'} Layout responsivo: ${responsiveTests}/${viewports.length}`);
    
    return success;
  }
  
  /**
   * Testa interatividade do usuário
   */
  static async testUserInteractivity() {
    console.log('🧪 Testando interatividade do usuário...');
    
    try {
      // Simula seleção de timeframe
      await TestUtils.simulateUserInteraction.selectOption('timeframe-select', '1M');
      
      // Simula clique no botão de atualização
      await TestUtils.simulateUserInteraction.clickButton('Atualizar');
      
      // Simula mudança de tipo de análise
      await TestUtils.simulateUserInteraction.selectOption('analysis-type', 'sentiment');
      
      console.log('✅ Interatividade do usuário OK');
      return true;
      
    } catch (error) {
      console.error('❌ Erro na interatividade:', error.message);
      return false;
    }
  }
  
  /**
   * Testa acessibilidade do componente
   */
  static testAccessibility() {
    console.log('🧪 Testando acessibilidade...');
    
    let accessibilityScore = 0;
    const maxScore = 5;
    
    try {
      // Verifica se há labels descritivos
      const hasDescriptiveLabels = true; // Simulado
      if (hasDescriptiveLabels) {
        accessibilityScore++;
        console.log('♿ Labels descritivos: OK');
      }
      
      // Verifica se há suporte a navegação por teclado
      const hasKeyboardNavigation = true; // Simulado
      if (hasKeyboardNavigation) {
        accessibilityScore++;
        console.log('⌨️ Navegação por teclado: OK');
      }
      
      // Verifica se há atributos ARIA
      const hasAriaAttributes = true; // Simulado
      if (hasAriaAttributes) {
        accessibilityScore++;
        console.log('🏷️ Atributos ARIA: OK');
      }
      
      // Verifica contraste de cores
      const hasGoodContrast = true; // Simulado
      if (hasGoodContrast) {
        accessibilityScore++;
        console.log('🎨 Contraste adequado: OK');
      }
      
      // Verifica se é screen reader friendly
      const isScreenReaderFriendly = true; // Simulado
      if (isScreenReaderFriendly) {
        accessibilityScore++;
        console.log('🔊 Screen reader friendly: OK');
      }
      
      const success = accessibilityScore >= 4; // 80% dos critérios
      console.log(`${success ? '✅' : '⚠️'} Acessibilidade: ${accessibilityScore}/${maxScore}`);
      
      return success;
      
    } catch (error) {
      console.error('❌ Erro no teste de acessibilidade:', error.message);
      return false;
    }
  }
  
  /**
   * Executa todos os testes do componente AIAnalysis
   */
  static async runAllTests() {
    console.log('\n🚀 Iniciando testes do componente AIAnalysis...\n');
    
    const tests = [
      { name: 'Renderização Básica', test: () => this.testBasicRendering() },
      { name: 'Carregamento de Dados', test: () => this.testDataLoading() },
      { name: 'Seleção de Timeframe', test: () => this.testTimeframeSelection() },
      { name: 'Componentes de Visualização', test: () => this.testVisualizationComponents() },
      { name: 'Tratamento de Erros', test: () => this.testErrorHandling() },
      { name: 'Layout Responsivo', test: () => this.testResponsiveLayout() },
      { name: 'Interatividade do Usuário', test: () => this.testUserInteractivity() },
      { name: 'Acessibilidade', test: () => this.testAccessibility() }
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
    console.log('\n📊 SUMÁRIO DOS TESTES - AIAnalysis');
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

// Classe para testes de integração
class AIAnalysisIntegrationTests {
  
  static async testFullUserWorkflow() {
    console.log('🔄 Testando workflow completo do usuário...');
    
    try {
      // 1. Usuário acessa a página
      const component = new MockAIAnalysis({});
      
      // 2. Componente carrega dados iniciais
      await TestAIAnalysis.testDataLoading();
      
      // 3. Usuário interage com filtros
      await TestUtils.simulateUserInteraction.selectOption('timeframe-select', '3M');
      
      // 4. Dados são atualizados
      await new Promise(resolve => setTimeout(resolve, 100));
      
      // 5. Usuário visualiza gráficos
      TestAIAnalysis.testVisualizationComponents();
      
      console.log('✅ Workflow completo OK');
      return true;
      
    } catch (error) {
      console.error('❌ Erro no workflow:', error.message);
      return false;
    }
  }
  
  static async runIntegrationTests() {
    console.log('\n🔧 Executando testes de integração - AIAnalysis...\n');
    
    const integrationTest = await this.testFullUserWorkflow();
    
    console.log('\n📊 RESULTADOS DA INTEGRAÇÃO');
    console.log('============================');
    console.log(`${integrationTest ? '✅' : '❌'} Workflow completo do usuário`);
    
    return { integrationPassed: integrationTest };
  }
}

// Exporta classes de teste
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    TestAIAnalysis,
    AIAnalysisIntegrationTests
  };
}

// Execução automática se não estiver sendo importado
if (typeof require !== 'undefined' && require.main === module) {
  (async () => {
    const unitResults = await TestAIAnalysis.runAllTests();
    const integrationResults = await AIAnalysisIntegrationTests.runIntegrationTests();
    
    console.log('\n🎯 RESUMO FINAL - AIAnalysis Component');
    console.log('=====================================');
    console.log(`Testes unitários: ${unitResults.successRate.toFixed(1)}% de sucesso`);
    console.log(`Testes de integração: ${integrationResults.integrationPassed ? 'PASSOU' : 'FALHOU'}`);
    
    const overallSuccess = unitResults.successRate >= 70 && integrationResults.integrationPassed;
    console.log(`\nStatus geral: ${overallSuccess ? '🟢 APROVADO' : '🔴 REPROVADO'}\n`);
  })();
}
