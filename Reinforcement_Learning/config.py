"""
Configuration pour l'environnement Grid World
Permet de configurer dynamiquement la taille de la grille, obstacles, goals, etc.
"""

class GridWorldConfig:
    """Configuration pour Grid World"""
    
    def __init__(
        self,
        grid_size=(10, 10),
        agent_start=(0, 0),
        goals=None,
        obstacles=None,
        step_reward=-0.01,
        goal_reward=1.0,
        obstacle_penalty=-1.0,
        max_steps=100
    ):
        """
        Args:
            grid_size: Tuple (height, width) de la grille
            agent_start: Position de départ de l'agent (row, col)
            goals: Liste de positions de goals [(row, col), ...]
            obstacles: Liste de positions d'obstacles [(row, col), ...]
            step_reward: Récompense pour chaque step
            goal_reward: Récompense pour atteindre un goal
            obstacle_penalty: Pénalité pour toucher un obstacle
            max_steps: Nombre maximum de steps par épisode
        """
        self.grid_size = grid_size
        self.height, self.width = grid_size
        self.agent_start = agent_start
        
        # Configuration par défaut si non spécifié
        if goals is None:
            self.goals = [(self.height - 1, self.width - 1)]
        else:
            self.goals = goals
            
        if obstacles is None:
            # Obstacles par défaut
            self.obstacles = [
                (self.height // 2, self.width // 3),
                (self.height // 2, 2 * self.width // 3)
            ]
        else:
            self.obstacles = obstacles
            
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.obstacle_penalty = obstacle_penalty
        self.max_steps = max_steps
        
    def validate(self):
        """Valide la configuration"""
        # Vérifier que l'agent n'est pas sur un obstacle ou goal
        if self.agent_start in self.obstacles:
            raise ValueError(f"Agent start position {self.agent_start} est sur un obstacle")
        
        # Vérifier que les positions sont dans la grille
        all_positions = [self.agent_start] + self.goals + self.obstacles
        for pos in all_positions:
            if not (0 <= pos[0] < self.height and 0 <= pos[1] < self.width):
                raise ValueError(f"Position {pos} en dehors de la grille")
        
        return True


# Configurations prédéfinies
DEFAULT_CONFIG = GridWorldConfig()

SMALL_CONFIG = GridWorldConfig(
    grid_size=(5, 5),
    agent_start=(0, 0),
    goals=[(4, 4)],
    obstacles=[(2, 2)]
)

LARGE_CONFIG = GridWorldConfig(
    grid_size=(15, 15),
    agent_start=(0, 0),
    goals=[(14, 14), (14, 0)],
    obstacles=[
        (7, 3), (7, 4), (7, 5),
        (7, 9), (7, 10), (7, 11)
    ]
)

COMPLEX_CONFIG = GridWorldConfig(
    grid_size=(12, 12),
    agent_start=(0, 0),
    goals=[(11, 11), (0, 11), (11, 0)],
    obstacles=[
        (5, 3), (5, 4), (5, 5), (5, 6),
        (6, 6), (7, 6), (8, 6),
        (3, 8), (4, 8), (5, 8)
    ]
)
