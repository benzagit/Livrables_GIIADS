"""
Environnement Grid World compatible avec Gymnasium
Permet d'utiliser l'environnement avec l'API standard de Gymnasium
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
from config import GridWorldConfig, DEFAULT_CONFIG


class GridWorldGymnasium(gym.Env):
    """
    Environnement Grid World compatible Gymnasium
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    
    # Actions possibles
    ACTIONS = {
        0: (-1, 0),  # UP
        1: (0, 1),   # RIGHT
        2: (1, 0),   # DOWN
        3: (0, -1)   # LEFT
    }
    
    ACTION_NAMES = {0: "UP", 1: "RIGHT", 2: "DOWN", 3: "LEFT"}
    
    def __init__(self, config=None, render_mode=None):
        """
        Args:
            config: GridWorldConfig object
            render_mode: Mode de rendu ("human" ou "rgb_array")
        """
        super().__init__()
        
        if config is None:
            config = DEFAULT_CONFIG
            
        config.validate()
        self.config = config
        
        self.height = config.height
        self.width = config.width
        self.goals = set(config.goals)
        self.obstacles = set(config.obstacles)
        self.start_pos = config.agent_start
        
        # Espaces d'observation et d'action (Gymnasium)
        # Observation: position (row, col)
        self.observation_space = spaces.Box(
            low=np.array([0, 0]), 
            high=np.array([self.height - 1, self.width - 1]),
            dtype=np.int32
        )
        
        # Actions: 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
        self.action_space = spaces.Discrete(4)
        
        # État interne
        self.agent_pos = None
        self.steps = 0
        
        # Rendering
        self.render_mode = render_mode
        self.fig = None
        self.ax = None
        self.value_states = None
        self.policy = None
        
        self.reset()
    
    def _get_obs(self):
        """Retourne l'observation courante"""
        return np.array(self.agent_pos, dtype=np.int32)
    
    def _get_info(self):
        """Retourne les informations additionnelles"""
        return {
            "agent_pos": self.agent_pos,
            "steps": self.steps,
            "distance_to_closest_goal": self._distance_to_closest_goal()
        }
    
    def _distance_to_closest_goal(self):
        """Calcule la distance Manhattan au goal le plus proche"""
        if not self.goals:
            return 0
        distances = [abs(self.agent_pos[0] - g[0]) + abs(self.agent_pos[1] - g[1]) 
                    for g in self.goals]
        return min(distances)
    
    def reset(self, seed=None, options=None):
        """
        Réinitialise l'environnement
        
        Args:
            seed: Graine pour le générateur aléatoire
            options: Options additionnelles
            
        Returns:
            observation: Observation initiale
            info: Informations additionnelles
        """
        super().reset(seed=seed)
        
        self.agent_pos = self.start_pos
        self.steps = 0
        
        observation = self._get_obs()
        info = self._get_info()
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, info
    
    def set_render_params(self, value_states=None, policy=None):
        """
        Configure les paramètres de rendu pour afficher value states et policy
        
        Args:
            value_states: Dictionnaire {state: value}
            policy: Dictionnaire {state: action}
        """
        self.value_states = value_states
        self.policy = policy
    
    def is_valid_position(self, pos):
        """Vérifie si une position est valide"""
        row, col = pos
        if not (0 <= row < self.height and 0 <= col < self.width):
            return False
        if pos in self.obstacles:
            return False
        return True
    
    def step(self, action):
        """
        Exécute une action
        
        Args:
            action: Action à exécuter (0-3)
            
        Returns:
            observation: Nouvelle observation
            reward: Récompense
            terminated: Si l'épisode est terminé (goal atteint)
            truncated: Si l'épisode est tronqué (timeout)
            info: Informations additionnelles
        """
        self.steps += 1
        
        # Calculer la nouvelle position
        delta = self.ACTIONS[action]
        new_pos = (self.agent_pos[0] + delta[0], self.agent_pos[1] + delta[1])
        
        # Vérifier si la nouvelle position est valide
        if self.is_valid_position(new_pos):
            self.agent_pos = new_pos
            reward = self.config.step_reward
        else:
            # Reste en place si mouvement invalide
            reward = self.config.obstacle_penalty
        
        # Vérifier si on a atteint un goal
        terminated = False
        if self.agent_pos in self.goals:
            reward = self.config.goal_reward
            terminated = True
        
        # Vérifier le timeout
        truncated = False
        if self.steps >= self.config.max_steps:
            truncated = True
        
        observation = self._get_obs()
        info = self._get_info()
        
        if terminated:
            info['success'] = True
        
        if self.render_mode == "human":
            self._render_frame()
        
        return observation, reward, terminated, truncated, info
    
    def render(self):
        """Rendu de l'environnement"""
        if self.render_mode == "rgb_array":
            return self._render_frame(self.value_states, self.policy)
        elif self.render_mode == "human":
            self._render_frame(self.value_states, self.policy)
    
    def _render_frame(self, value_states=None, policy=None):
        """Génère une frame de rendu"""
        if self.fig is None:
            plt.ion()
            self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        self.ax.clear()
        
        # Créer la grille
        for i in range(self.height + 1):
            self.ax.plot([0, self.width], [i, i], 'k-', linewidth=0.5)
        for j in range(self.width + 1):
            self.ax.plot([j, j], [0, self.height], 'k-', linewidth=0.5)
        
        # Afficher les value states si fournis
        if value_states is not None:
            from matplotlib.patches import Rectangle
            values = [v for v in value_states.values() if v != float('-inf')]
            if values:
                vmin, vmax = min(values), max(values)
                
                for state, value in value_states.items():
                    if value != float('-inf') and state not in self.obstacles:
                        row, col = state
                        if vmax > vmin:
                            normalized = (value - vmin) / (vmax - vmin)
                        else:
                            normalized = 0.5
                        color = plt.cm.RdYlGn(normalized)
                        
                        rect = Rectangle((col, self.height - row - 1), 1, 1, 
                                       facecolor=color, alpha=0.6)
                        self.ax.add_patch(rect)
                        
                        # Afficher la valeur
                        self.ax.text(col + 0.5, self.height - row - 0.5, 
                                  f'{value:.2f}',
                                  ha='center', va='center', fontsize=8, 
                                  color='black', weight='bold')
        
        # Afficher la politique si fournie
        if policy is not None:
            arrow_props = dict(arrowstyle='->', lw=2, color='darkblue')
            for state, action in policy.items():
                if state not in self.obstacles and state not in self.goals:
                    row, col = state
                    delta = self.ACTIONS[action]
                    x = col + 0.5
                    y = self.height - row - 0.5
                    dx = delta[1] * 0.3
                    dy = -delta[0] * 0.3
                    
                    self.ax.annotate('', xy=(x + dx, y + dy), xytext=(x, y),
                              arrowprops=arrow_props)
        
        # Dessiner les obstacles
        for obs in self.obstacles:
            row, col = obs
            from matplotlib.patches import Rectangle
            rect = Rectangle((col, self.height - row - 1), 1, 1, 
                           facecolor='black', alpha=0.8)
            self.ax.add_patch(rect)
            self.ax.text(col + 0.5, self.height - row - 0.5, 'X',
                   ha='center', va='center', fontsize=16, color='white', weight='bold')
        
        # Dessiner les goals
        for goal in self.goals:
            row, col = goal
            rect = Rectangle((col, self.height - row - 1), 1, 1, 
                           facecolor='gold', alpha=0.8)
            self.ax.add_patch(rect)
            self.ax.text(col + 0.5, self.height - row - 0.5, 'G',
                   ha='center', va='center', fontsize=16, color='black', weight='bold')
        
        # Dessiner l'agent
        row, col = self.agent_pos
        circle = plt.Circle((col + 0.5, self.height - row - 0.5), 0.3, 
                          color='blue', alpha=0.8)
        self.ax.add_patch(circle)
        self.ax.text(col + 0.5, self.height - row - 0.5, 'A',
               ha='center', va='center', fontsize=12, color='white', weight='bold')
        
        self.ax.set_xlim(0, self.width)
        self.ax.set_ylim(0, self.height)
        self.ax.set_aspect('equal')
        self.ax.invert_yaxis()
        
        title = f'Grid World (Gymnasium) - Step: {self.steps}'
        if value_states is not None:
            title += ' - Value States'
        if policy is not None:
            title += ' + Policy'
        self.ax.set_title(title, fontsize=14, weight='bold')
        
        if self.render_mode == "human":
            plt.pause(0.001)
            plt.draw()
        
        # Pour rgb_array, convertir la figure en array
        if self.render_mode == "rgb_array":
            self.fig.canvas.draw()
            data = np.frombuffer(self.fig.canvas.tostring_rgb(), dtype=np.uint8)
            data = data.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))
            return data
    
    def close(self):
        """Ferme l'environnement"""
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


def test_gymnasium_environment():
    """Teste l'environnement Gymnasium"""
    print("=" * 60)
    print("TESTING GYMNASIUM ENVIRONMENT")
    print("=" * 60)
    
    # Créer l'environnement
    env = GridWorldGymnasium(config=DEFAULT_CONFIG, render_mode="human")
    
    print(f"\nObservation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Grid Size: {env.height}x{env.width}")
    print(f"Start Position: {env.start_pos}")
    print(f"Goals: {env.goals}")
    print(f"Obstacles: {env.obstacles}")
    
    # Test: épisode aléatoire
    print("\n" + "=" * 60)
    print("Running random episode...")
    print("=" * 60)
    
    observation, info = env.reset()
    print(f"Initial observation: {observation}")
    print(f"Initial info: {info}")
    
    total_reward = 0
    terminated = False
    truncated = False
    step = 0
    
    while not (terminated or truncated):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step += 1
        
        print(f"Step {step}: Action={env.ACTION_NAMES[action]}, "
              f"Observation={observation}, Reward={reward:.3f}, "
              f"Terminated={terminated}, Truncated={truncated}")
        
        plt.pause(0.3)
    
    print(f"\nEpisode finished!")
    print(f"Total Reward: {total_reward:.3f}")
    print(f"Steps: {step}")
    print(f"Success: {info.get('success', False)}")
    
    # Fermer l'environnement
    env.close()
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    test_gymnasium_environment()
