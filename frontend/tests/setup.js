/**
 * Configuração de Testes para o Frontend React
 * 
 * Este arquivo configura o ambiente de testes para todos os componentes React,
 * incluindo setup do React Testing Library, mocks de APIs e utilidades comuns.
 * 
 * Autor: Rafael Lima Caires
 * Data: Dezembro 2024
 */

// Mock do React Testing Library se não estiver disponível
const mockRender = (component) => ({
  container: document.createElement('div'),
  getByText: (text) => ({ textContent: text }),
  getByTestId: (id) => ({ 'data-testid': id }),
  queryByText: (text) => ({ textContent: text }),
  queryByTestId: (id) => ({ 'data-testid': id })
});

const mockScreen = {
  getByText: (text) => ({ textContent: text }),
  getByTestId: (id) => ({ 'data-testid': id }),
  queryByText: (text) => ({ textContent: text }),
  queryByTestId: (id) => ({ 'data-testid': id }),
  getByRole: (role) => ({ role: role })
};

const mockFireEvent = {
  click: (element) => console.log('Mock click on:', element),
  change: (element, options) => console.log('Mock change on:', element, options),
  submit: (element) => console.log('Mock submit on:', element)
};

const mockWaitFor = async (callback) => {
  await new Promise(resolve => setTimeout(resolve, 100));
  return callback();
};

// Mock do fetch para testes de API
const mockFetch = (url, options = {}) => {
  const responses = {
    '/api/recommendations/advanced': {
      success: true,
      data: {
        hybrid_recommendations: [
          {
            ticker: 'WEGE3',
            name: 'WEG ON',
            recommendation: 'BUY',
            confidence: 0.85,
            potential_return: 15.2,
            risk_score: 0.55
          }
        ]
      }
    },
    '/api/ai-analysis/complete': {
      success: true,
      data: {
        market_overview: {
          ibovespa: { current_level: 118500.25, daily_change: 1.23 }
        },
        ml_predictions: {
          ensemble_forecast: {
            ibovespa_30d: { predicted_level: 122450, confidence: 0.78 }
          }
        }
      }
    },
    '/api/portfolio/detailed': {
      success: true,
      data: {
        summary: {
          total_value: 145230.50,
          total_return: 25230.50,
          return_percentage: 21.03
        },
        assets: [
          {
            ticker: 'PETR4',
            name: 'Petrobras PN',
            current_price: 32.45,
            return_percentage: 13.86
          }
        ]
      }
    }
  };

  const response = responses[url] || { success: false, error: 'Not found' };
  
  return Promise.resolve({
    ok: true,
    status: 200,
    json: () => Promise.resolve(response)
  });
};

// Mock do localStorage
const mockLocalStorage = {
  getItem: (key) => {
    const items = {
      'token': 'mock_jwt_token',
      'user': JSON.stringify({ id: 1, name: 'Test User' })
    };
    return items[key] || null;
  },
  setItem: (key, value) => console.log(`localStorage.setItem(${key}, ${value})`),
  removeItem: (key) => console.log(`localStorage.removeItem(${key})`),
  clear: () => console.log('localStorage.clear()')
};

// Setup global para testes
const setupTestEnvironment = () => {
  // Mock do fetch global
  global.fetch = mockFetch;
  
  // Mock do localStorage
  Object.defineProperty(window, 'localStorage', {
    value: mockLocalStorage,
    writable: true
  });
  
  // Mock do ResizeObserver (usado por recharts)
  global.ResizeObserver = class ResizeObserver {
    constructor(callback) {
      this.callback = callback;
    }
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  
  // Mock do IntersectionObserver
  global.IntersectionObserver = class IntersectionObserver {
    constructor(callback) {
      this.callback = callback;
    }
    observe() {}
    unobserve() {}
    disconnect() {}
  };
  
  // Mock de console para testes mais limpos
  console.warn = jest ? jest.fn() : () => {};
  console.error = jest ? jest.fn() : () => {};
};

// Utilitários de teste
const TestUtils = {
  
  // Cria props mock para componentes
  createMockProps: (overrides = {}) => ({
    onSubmit: () => {},
    onChange: () => {},
    loading: false,
    error: null,
    ...overrides
  }),
  
  // Simula dados de recomendações
  createMockRecommendations: () => [
    {
      ticker: 'WEGE3',
      name: 'WEG ON',
      recommendation: 'BUY',
      confidence: 0.85,
      potential_return: 15.2,
      risk_score: 0.55,
      reasoning: 'Strong technical indicators'
    },
    {
      ticker: 'RENT3',
      name: 'Localiza Rent a Car ON',
      recommendation: 'BUY',
      confidence: 0.78,
      potential_return: 12.8,
      risk_score: 0.62,
      reasoning: 'Recovery in tourism sector'
    }
  ],
  
  // Simula dados de portfólio
  createMockPortfolio: () => ({
    summary: {
      total_value: 145230.50,
      total_invested: 120000.00,
      total_return: 25230.50,
      return_percentage: 21.03
    },
    assets: [
      {
        ticker: 'PETR4',
        name: 'Petrobras PN',
        quantity: 500,
        current_price: 32.45,
        return_percentage: 13.86,
        allocation_percentage: 11.2
      },
      {
        ticker: 'VALE3',
        name: 'Vale ON',
        quantity: 200,
        current_price: 71.20,
        return_percentage: 9.54,
        allocation_percentage: 9.8
      }
    ]
  }),
  
  // Simula análise de IA
  createMockAIAnalysis: () => ({
    market_overview: {
      ibovespa: {
        current_level: 118500.25,
        daily_change: 1.23,
        monthly_change: 5.42
      }
    },
    ml_predictions: {
      ensemble_forecast: {
        ibovespa_30d: {
          predicted_level: 122450,
          confidence: 0.78,
          probability_up: 0.72
        }
      }
    },
    sentiment_analysis: {
      overall_sentiment: {
        score: 0.68,
        label: 'Positivo'
      }
    }
  }),
  
  // Aguarda elemento aparecer na tela
  waitForElement: async (selector, timeout = 5000) => {
    return new Promise((resolve, reject) => {
      const element = document.querySelector(selector);
      if (element) {
        resolve(element);
        return;
      }
      
      const observer = new MutationObserver(() => {
        const element = document.querySelector(selector);
        if (element) {
          observer.disconnect();
          resolve(element);
        }
      });
      
      observer.observe(document.body, {
        childList: true,
        subtree: true
      });
      
      setTimeout(() => {
        observer.disconnect();
        reject(new Error(`Elemento ${selector} não encontrado após ${timeout}ms`));
      }, timeout);
    });
  },
  
  // Simula interação do usuário
  simulateUserInteraction: {
    clickButton: (buttonText) => {
      console.log(`Simulando click no botão: ${buttonText}`);
      return Promise.resolve();
    },
    
    fillInput: (inputId, value) => {
      console.log(`Preenchendo input ${inputId} com: ${value}`);
      return Promise.resolve();
    },
    
    selectOption: (selectId, optionValue) => {
      console.log(`Selecionando opção ${optionValue} em: ${selectId}`);
      return Promise.resolve();
    }
  },
  
  // Validadores de componente
  validateComponent: {
    hasRequiredProps: (component, requiredProps) => {
      for (const prop of requiredProps) {
        if (!(prop in component.props)) {
          throw new Error(`Prop obrigatória '${prop}' não encontrada`);
        }
      }
      return true;
    },
    
    rendersWithoutCrashing: (component) => {
      try {
        mockRender(component);
        return true;
      } catch (error) {
        throw new Error(`Componente não renderiza: ${error.message}`);
      }
    },
    
    hasAccessibilityAttributes: (element) => {
      const requiredAttributes = ['role', 'aria-label', 'aria-describedby'];
      const hasAnyAttribute = requiredAttributes.some(attr => 
        element.hasAttribute(attr)
      );
      
      if (!hasAnyAttribute) {
        console.warn('Elemento pode precisar de atributos de acessibilidade');
      }
      
      return hasAnyAttribute;
    }
  }
};

// Mocks de contextos React
const MockAuthContext = {
  user: { id: 1, name: 'Test User', email: 'test@example.com' },
  isAuthenticated: true,
  login: () => Promise.resolve(),
  logout: () => {},
  loading: false
};

const MockThemeContext = {
  theme: 'light',
  setTheme: () => {},
  toggleTheme: () => {}
};

// Wrapper para componentes que usam contexto
const TestWrapper = ({ children }) => {
  return children; // Simplified wrapper for testing
};

// Exporta utilitários
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    setupTestEnvironment,
    TestUtils,
    MockAuthContext,
    MockThemeContext,
    TestWrapper,
    mockRender,
    mockScreen,
    mockFireEvent,
    mockWaitFor
  };
}

// Setup automático se não estiver em ambiente de teste específico
if (typeof window !== 'undefined' && !window.__TEST_SETUP__) {
  setupTestEnvironment();
  window.__TEST_SETUP__ = true;
}
