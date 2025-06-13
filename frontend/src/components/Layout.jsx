import { useAuth } from '../contexts/AuthContext'
import { useTheme } from '../contexts/ThemeContext'
import { Button } from './ui/button'
import { Avatar, AvatarFallback } from './ui/avatar'
import { 
  DropdownMenu, 
  DropdownMenuContent, 
  DropdownMenuItem, 
  DropdownMenuTrigger,
  DropdownMenuSeparator 
} from './ui/dropdown-menu'
import { 
  Sidebar, 
  SidebarContent, 
  SidebarHeader, 
  SidebarMenu, 
  SidebarMenuButton, 
  SidebarMenuItem,
  SidebarProvider,
  SidebarInset,
  SidebarTrigger
} from './ui/sidebar'
import { 
  TrendingUp, 
  BarChart3, 
  Target, 
  Briefcase, 
  User, 
  LogOut, 
  Sun, 
  Moon,
  ChevronDown,
  Brain
} from 'lucide-react'
import { Link, useLocation } from 'react-router-dom'

const menuItems = [
  {
    title: 'Dashboard',
    url: '/dashboard',
    icon: BarChart3
  },
  {
    title: 'Previsões',
    url: '/previsoes',
    icon: TrendingUp
  },
  {
    title: 'Recomendações',
    url: '/recomendacoes',
    icon: Target
  },
  {
    title: 'Portfólio',
    url: '/portfolio',
    icon: Briefcase
  },
  {
    title: 'Análise IA',
    url: '/ai-analysis',
    icon: Brain
  },
  {
    title: 'Perfil',
    url: '/perfil',
    icon: User
  }
]

export default function Layout({ children }) {
  const { user, logout } = useAuth()
  const { theme, toggleTheme } = useTheme()
  const location = useLocation()

  const handleLogout = () => {
    logout()
  }

  const getUserInitials = (name) => {
    return name
      .split(' ')
      .map(word => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2)
  }

  const getRiskProfileColor = (profile) => {
    switch (profile) {
      case 'conservador':
        return 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200'
      case 'moderado':
        return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900 dark:text-yellow-200'
      case 'arrojado':
        return 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200'
      default:
        return 'bg-gray-100 text-gray-800 dark:bg-gray-900 dark:text-gray-200'
    }
  }

  const getRiskProfileLabel = (profile) => {
    switch (profile) {
      case 'conservador':
        return 'Conservador'
      case 'moderado':
        return 'Moderado'
      case 'arrojado':
        return 'Arrojado'
      default:
        return profile
    }
  }

  return (
    <SidebarProvider>
      <Sidebar>
        <SidebarHeader>
          <div className="flex items-center gap-2 px-4 py-2">
            <div className="w-8 h-8 bg-blue-600 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-5 h-5 text-white" />
            </div>
            <span className="font-semibold text-lg">Sistema Financeiro</span>
          </div>
        </SidebarHeader>
        <SidebarContent>
          <SidebarMenu>
            {menuItems.map((item) => (
              <SidebarMenuItem key={item.title}>
                <SidebarMenuButton 
                  asChild 
                  isActive={location.pathname === item.url}
                >
                  <Link to={item.url}>
                    <item.icon className="w-4 h-4" />
                    <span>{item.title}</span>
                  </Link>
                </SidebarMenuButton>
              </SidebarMenuItem>
            ))}
          </SidebarMenu>
        </SidebarContent>
      </Sidebar>
      
      <SidebarInset>
        <header className="flex h-16 shrink-0 items-center gap-2 border-b px-4">
          <SidebarTrigger className="-ml-1" />
          
          <div className="ml-auto flex items-center gap-4">
            <Button
              variant="ghost"
              size="icon"
              onClick={toggleTheme}
            >
              {theme === 'dark' ? (
                <Sun className="h-4 w-4" />
              ) : (
                <Moon className="h-4 w-4" />
              )}
            </Button>
            
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" className="flex items-center gap-2">
                  <Avatar className="h-8 w-8">
                    <AvatarFallback>
                      {getUserInitials(user?.name || 'U')}
                    </AvatarFallback>
                  </Avatar>
                  <div className="flex flex-col items-start">
                    <span className="text-sm font-medium">{user?.name}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full ${getRiskProfileColor(user?.risk_profile)}`}>
                      {getRiskProfileLabel(user?.risk_profile)}
                    </span>
                  </div>
                  <ChevronDown className="h-4 w-4" />
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="end">
                <DropdownMenuItem asChild>
                  <Link to="/perfil">
                    <User className="mr-2 h-4 w-4" />
                    <span>Perfil</span>
                  </Link>
                </DropdownMenuItem>
                <DropdownMenuSeparator />
                <DropdownMenuItem onClick={handleLogout}>
                  <LogOut className="mr-2 h-4 w-4" />
                  <span>Sair</span>
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>
        </header>
        
        <main className="flex-1 overflow-auto p-6">
          {children}
        </main>
        
        <footer className="border-t px-6 py-4">
          <div className="flex items-center justify-between text-sm text-muted-foreground">
            <span>© 2025 Sistema Financeiro. Todos os direitos reservados.</span>
            <div className="flex gap-4">
              <span>Termos de Uso</span>
              <span>Política de Privacidade</span>
              <span>Contato</span>
            </div>
          </div>
        </footer>
      </SidebarInset>
    </SidebarProvider>
  )
}

