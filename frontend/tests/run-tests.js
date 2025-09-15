/**
 * Test Runner - Executa toda a suite de testes
 * 
 * Este arquivo orquestra a execução de todos os testes do sistema,
 * incluindo testes unitários, integração e end-to-end.
 * 
 * Autor: Rafael Lima Caires
 * Data: Dezembro 2024
 */

// Importa todos os módulos de teste
const { setupTestEnvironment } = require('./setup');

// Setup inicial
setupTestEnvironment();

// Importa classes de teste
let TestAIAnalysis, AIAnalysisIntegrationTests;
let TestRecommendations, RecommendationsPerformanceTests, RecommendationsIntegrationTests;
let TestPortfolio, PortfolioIntegrationTests;
let EndToEndTests;

try {
  ({ TestAIAnalysis, AIAnalysisIntegrationTests } = require('./test-aianalysis'));
  ({ TestRecommendations, RecommendationsPerformanceTests, RecommendationsIntegrationTests } = require('./test-recommendations'));
  ({ TestPortfolio, PortfolioIntegrationTests } = require('./test-portfolio'));
  ({ EndToEndTests } = require('./test-e2e'));
} catch (error) {
  console.warn('⚠️ Alguns módulos de teste podem não estar disponíveis:', error.message);
}

class TestRunner {
  
  constructor() {
    this.results = {
      unit: {},
      integration: {},
      e2e: {},
      performance: {}
    };
  }
  
  /**
   * Executa testes unitários de todos os componentes
   */
  async runUnitTests() {
    console.log('\n🧪 EXECUTANDO TESTES UNITÁRIOS');
    console.log('==============================\n');
    
    const unitResults = {};
    
    // Testes do AIAnalysis
    if (TestAIAnalysis) {
      console.log('🤖 Testando componente AIAnalysis...');
      try {
        unitResults.aiAnalysis = await TestAIAnalysis.runAllTests();
      } catch (error) {
        console.error('❌ Erro nos testes do AIAnalysis:', error.message);
        unitResults.aiAnalysis = { successRate: 0, error: error.message };
      }
    }
    
    // Testes do Recommendations
    if (TestRecommendations) {
      console.log('\n💡 Testando componente Recommendations...');
      try {
        unitResults.recommendations = await TestRecommendations.runAllTests();
      } catch (error) {
        console.error('❌ Erro nos testes do Recommendations:', error.message);
        unitResults.recommendations = { successRate: 0, error: error.message };
      }
    }
    
    // Testes do Portfolio
    if (TestPortfolio) {
      console.log('\n💼 Testando componente Portfolio...');
      try {
        unitResults.portfolio = await TestPortfolio.runAllTests();
      } catch (error) {
        console.error('❌ Erro nos testes do Portfolio:', error.message);
        unitResults.portfolio = { successRate: 0, error: error.message };
      }
    }
    
    this.results.unit = unitResults;
    return unitResults;
  }
  
  /**
   * Executa testes de integração
   */
  async runIntegrationTests() {
    console.log('\n🔧 EXECUTANDO TESTES DE INTEGRAÇÃO');
    console.log('==================================\n');
    
    const integrationResults = {};
    
    // Integração AIAnalysis
    if (AIAnalysisIntegrationTests) {
      console.log('🤖 Testando integração do AIAnalysis...');
      try {
        integrationResults.aiAnalysis = await AIAnalysisIntegrationTests.runIntegrationTests();
      } catch (error) {
        console.error('❌ Erro na integração do AIAnalysis:', error.message);
        integrationResults.aiAnalysis = { integrationPassed: false, error: error.message };
      }
    }
    
    // Integração Recommendations
    if (RecommendationsIntegrationTests) {
      console.log('\n💡 Testando integração do Recommendations...');
      try {
        integrationResults.recommendations = await RecommendationsIntegrationTests.runIntegrationTests();
      } catch (error) {
        console.error('❌ Erro na integração do Recommendations:', error.message);
        integrationResults.recommendations = { integrationPassed: false, error: error.message };
      }
    }
    
    // Integração Portfolio
    if (PortfolioIntegrationTests) {
      console.log('\n💼 Testando integração do Portfolio...');
      try {
        integrationResults.portfolio = await PortfolioIntegrationTests.runIntegrationTests();
      } catch (error) {
        console.error('❌ Erro na integração do Portfolio:', error.message);
        integrationResults.portfolio = { integrationPassed: false, error: error.message };
      }
    }
    
    this.results.integration = integrationResults;
    return integrationResults;
  }
  
  /**
   * Executa testes de performance
   */
  async runPerformanceTests() {
    console.log('\n⚡ EXECUTANDO TESTES DE PERFORMANCE');
    console.log('==================================\n');
    
    const performanceResults = {};
    
    // Performance Recommendations
    if (RecommendationsPerformanceTests) {
      console.log('💡 Testando performance do Recommendations...');
      try {
        performanceResults.recommendations = await RecommendationsPerformanceTests.runPerformanceTests();
      } catch (error) {
        console.error('❌ Erro na performance do Recommendations:', error.message);
        performanceResults.recommendations = { performancePassed: false, error: error.message };
      }
    }
    
    this.results.performance = performanceResults;
    return performanceResults;
  }
  
  /**
   * Executa testes end-to-end
   */
  async runE2ETests() {
    console.log('\n🎬 EXECUTANDO TESTES END-TO-END');
    console.log('==============================\n');
    
    let e2eResults = {};
    
    if (EndToEndTests) {
      console.log('🌐 Executando testes de sistema completo...');
      try {
        e2eResults = await EndToEndTests.runAllE2ETests();
      } catch (error) {
        console.error('❌ Erro nos testes E2E:', error.message);
        e2eResults = { overallSuccess: false, successRate: 0, error: error.message };
      }
    }
    
    this.results.e2e = e2eResults;
    return e2eResults;
  }
  
  /**
   * Gera relatório final consolidado
   */
  generateFinalReport() {
    console.log('\n📊 RELATÓRIO FINAL DOS TESTES');
    console.log('=============================\n');
    
    const report = {
      timestamp: new Date().toISOString(),
      summary: {
        unit: this.calculateUnitTestsSummary(),
        integration: this.calculateIntegrationSummary(),
        performance: this.calculatePerformanceSummary(),
        e2e: this.calculateE2ESummary()
      },
      details: this.results
    };
    
    // Exibe sumário executivo
    console.log('🎯 SUMÁRIO EXECUTIVO');
    console.log('===================');
    console.log(`📈 Testes Unitários: ${report.summary.unit.averageSuccess.toFixed(1)}%`);
    console.log(`🔧 Testes de Integração: ${report.summary.integration.overallSuccess ? 'PASSOU' : 'FALHOU'}`);
    console.log(`⚡ Testes de Performance: ${report.summary.performance.overallSuccess ? 'PASSOU' : 'FALHOU'}`);
    console.log(`🌐 Testes End-to-End: ${report.summary.e2e.success ? 'PASSOU' : 'FALHOU'} (${report.summary.e2e.successRate.toFixed(1)}%)`);
    
    // Determina status geral
    const overallQuality = this.determineOverallQuality(report.summary);
    
    console.log('\n🎖️ QUALIDADE GERAL DO SISTEMA');
    console.log('=============================');
    console.log(`Status: ${overallQuality.status}`);
    console.log(`Nota: ${overallQuality.grade}`);
    console.log(`Descrição: ${overallQuality.description}`);
    
    // Recomendações
    console.log('\n💡 RECOMENDAÇÕES');
    console.log('================');
    overallQuality.recommendations.forEach((rec, index) => {
      console.log(`${index + 1}. ${rec}`);
    });
    
    return report;
  }
  
  /**
   * Calcula sumário dos testes unitários
   */
  calculateUnitTestsSummary() {
    const unitResults = this.results.unit;
    const components = Object.keys(unitResults);
    
    if (components.length === 0) {
      return { averageSuccess: 0, componentsCount: 0 };
    }
    
    const totalSuccess = components.reduce((sum, component) => {
      return sum + (unitResults[component].successRate || 0);
    }, 0);
    
    return {
      averageSuccess: totalSuccess / components.length,
      componentsCount: components.length,
      details: unitResults
    };
  }
  
  /**
   * Calcula sumário dos testes de integração
   */
  calculateIntegrationSummary() {
    const integrationResults = this.results.integration;
    const components = Object.keys(integrationResults);
    
    const passedComponents = components.filter(component => 
      integrationResults[component].integrationPassed
    ).length;
    
    return {
      overallSuccess: passedComponents === components.length && components.length > 0,
      passedComponents,
      totalComponents: components.length,
      details: integrationResults
    };
  }
  
  /**
   * Calcula sumário dos testes de performance
   */
  calculatePerformanceSummary() {
    const performanceResults = this.results.performance;
    const components = Object.keys(performanceResults);
    
    const passedComponents = components.filter(component => 
      performanceResults[component].performancePassed
    ).length;
    
    return {
      overallSuccess: passedComponents === components.length && components.length > 0,
      passedComponents,
      totalComponents: components.length,
      details: performanceResults
    };
  }
  
  /**
   * Calcula sumário dos testes E2E
   */
  calculateE2ESummary() {
    const e2eResults = this.results.e2e;
    
    return {
      success: e2eResults.overallSuccess || false,
      successRate: e2eResults.successRate || 0,
      totalTests: e2eResults.total || 0,
      passedTests: e2eResults.passed || 0,
      details: e2eResults
    };
  }
  
  /**
   * Determina qualidade geral do sistema
   */
  determineOverallQuality(summary) {
    const unitScore = summary.unit.averageSuccess;
    const integrationScore = summary.integration.overallSuccess ? 100 : 0;
    const performanceScore = summary.performance.overallSuccess ? 100 : 0;
    const e2eScore = summary.e2e.successRate;
    
    // Calcula nota ponderada
    const weightedScore = (
      unitScore * 0.35 +           // 35% - Testes unitários
      integrationScore * 0.25 +    // 25% - Testes de integração
      performanceScore * 0.15 +    // 15% - Testes de performance
      e2eScore * 0.25             // 25% - Testes E2E
    );
    
    let status, grade, description, recommendations;
    
    if (weightedScore >= 90) {
      status = '🟢 EXCELENTE';
      grade = 'A+';
      description = 'Sistema de altíssima qualidade, pronto para produção.';
      recommendations = [
        'Sistema aprovado para produção',
        'Manter rotina de testes regulares',
        'Considerar implementar testes automatizados em CI/CD'
      ];
    } else if (weightedScore >= 80) {
      status = '🟢 BOM';
      grade = 'A';
      description = 'Sistema de boa qualidade com pequenos pontos de melhoria.';
      recommendations = [
        'Sistema aprovado para produção',
        'Corrigir pontos menores identificados',
        'Melhorar cobertura de testes onde necessário'
      ];
    } else if (weightedScore >= 70) {
      status = '🟡 SATISFATÓRIO';
      grade = 'B';
      description = 'Sistema funcional mas necessita melhorias importantes.';
      recommendations = [
        'Implementar melhorias antes da produção',
        'Focar nos testes que falharam',
        'Revisar arquitetura dos componentes críticos'
      ];
    } else if (weightedScore >= 60) {
      status = '🟠 PRECISA MELHORAR';
      grade = 'C';
      description = 'Sistema com problemas significativos que impedem produção.';
      recommendations = [
        'NÃO aprovar para produção',
        'Implementar correções extensivas',
        'Revisar design e arquitetura do sistema',
        'Executar nova rodada de testes após correções'
      ];
    } else {
      status = '🔴 CRÍTICO';
      grade = 'D';
      description = 'Sistema com falhas graves que impedem uso seguro.';
      recommendations = [
        'SISTEMA REPROVADO para produção',
        'Revisar completamente a implementação',
        'Considerar refatoração major',
        'Implementar plano de correção abrangente'
      ];
    }
    
    return {
      status,
      grade,
      description,
      score: weightedScore,
      recommendations
    };
  }
  
  /**
   * Executa todos os testes em sequência
   */
  async runAllTests() {
    console.log('\n🚀 INICIANDO EXECUÇÃO COMPLETA DOS TESTES');
    console.log('=========================================');
    console.log(`⏰ Início: ${new Date().toLocaleString()}\n`);
    
    const startTime = performance.now();
    
    try {
      // 1. Testes unitários
      await this.runUnitTests();
      
      // 2. Testes de integração
      await this.runIntegrationTests();
      
      // 3. Testes de performance
      await this.runPerformanceTests();
      
      // 4. Testes end-to-end
      await this.runE2ETests();
      
      // 5. Relatório final
      const report = this.generateFinalReport();
      
      const endTime = performance.now();
      const totalDuration = ((endTime - startTime) / 1000).toFixed(2);
      
      console.log(`\n⏱️ TEMPO TOTAL DE EXECUÇÃO: ${totalDuration} segundos`);
      console.log(`🏁 Fim: ${new Date().toLocaleString()}`);
      
      return report;
      
    } catch (error) {
      console.error('\n💥 ERRO CRÍTICO NA EXECUÇÃO DOS TESTES');
      console.error('=====================================');
      console.error('Erro:', error.message);
      console.error('Stack:', error.stack);
      
      return {
        error: true,
        message: error.message,
        results: this.results
      };
    }
  }
}

// Função para executar testes se chamado diretamente
async function main() {
  const runner = new TestRunner();
  const results = await runner.runAllTests();
  
  // Salva relatório em arquivo se possível
  if (typeof require !== 'undefined') {
    const fs = require('fs');
    const path = require('path');
    
    try {
      const reportPath = path.join(__dirname, 'test-report.json');
      fs.writeFileSync(reportPath, JSON.stringify(results, null, 2));
      console.log(`\n📝 Relatório salvo em: ${reportPath}`);
    } catch (error) {
      console.warn('⚠️ Não foi possível salvar o relatório:', error.message);
    }
  }
  
  return results;
}

// Exporta classes e funções
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    TestRunner,
    main
  };
}

// Executa se chamado diretamente
if (typeof require !== 'undefined' && require.main === module) {
  main().then(() => {
    console.log('\n✨ Execução dos testes finalizada!');
    process.exit(0);
  }).catch(error => {
    console.error('\n💥 Erro fatal:', error);
    process.exit(1);
  });
}
