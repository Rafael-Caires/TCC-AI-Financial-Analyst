import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import tailwindcss from '@tailwindcss/vite'
import path from 'path'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: {
      "@": path.resolve(__dirname, "./src"),
    },
  },
  // Adicione a configuração do proxy aqui
  server: {
    proxy: {
      // Qualquer requisição que comece com '/api' será redirecionada
      '/api': {
        target: 'http://localhost:5000', // O endereço do seu backend
        changeOrigin: true,              // Essencial para o proxy funcionar corretamente
        secure: false,                   // Defina como false se seu backend roda em http
      },
    },
  },
})