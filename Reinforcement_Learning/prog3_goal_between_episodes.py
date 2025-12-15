"""
PROGRAMME 3: Value Iteration avec Goal qui se déplace ENTRE les épisodes
Le goal change de position à chaque nouvel épisode
"""

import numpy as np
import matplotlib.pyplot as plt
from gym_environment import GridWorldGymnasium
from config import GridWorldConfig


class ValueIterationAgent:
    """Agent avec Value Iteration"""
    
    def __init__(self, env, gamma=0.9, theta=1e-6):
        self.env = env
        self.gamma = gamma
        self.theta = theta
        
        # États (toutes positions valides)
        self.states = []
        for row in range(env.height):
            for col in range(env.width):
                if (row, col) not in env.obstacles:
                    self.states.append((row, col))
        
        # Initialisation
        self.V = {state: 0.0 for state in self.states}
        self.policy = {}
    
    def get_transition(self, state, action):
        """Simule une transition"""
        delta = self.env.ACTIONS[action]
        new_pos = (state[0] + delta[0], state[1] + delta[1])
        
        if self.env.is_valid_position(new_pos):
            next_state = new_pos
            reward = self.env.config.step_reward
        else:
            next_state = state
            reward = self.env.config.obstacle_penalty
        
        terminated = next_state in self.env.goals
        if terminated:
            reward = self.env.config.goal_reward
        
        return next_state, reward, terminated
    
    def value_iteration(self, verbose=False):
        """Algorithme Value Iteration"""
        if verbose:
            print("🔄 Réentraînement avec nouveau goal...")
        
        # Réinitialiser les valeurs
        self.V = {state: 0.0 for state in self.states}
        
        for iteration in range(1000):
            delta = 0
            
            for state in self.states:
                if state in self.env.goals:
                    continue
                
                old_value = self.V[state]
                
                # Max sur toutes les actions
                action_values = []
                for action in range(self.env.action_space.n):
                    next_state, reward, terminated = self.get_transition(state, action)
                    value = reward + (0 if terminated else self.gamma * self.V[next_state])
                    action_values.append(value)
                
                self.V[state] = max(action_values)
                delta = max(delta, abs(old_value - self.V[state]))
            
            if verbose and (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: Delta={delta:.6f}")
            
            if delta < self.theta:
                if verbose:
                    print(f"✅ Convergence en {iteration + 1} itérations!\n")
                return iteration + 1
        
        return 1000
    
    def extract_policy(self):
        """Extrait la politique optimale"""
        self.policy = {}
        for state in self.states:
            if state in self.env.goals:
                continue
            
            best_action = 0
            best_value = float('-inf')
            
            for action in range(self.env.action_space.n):
                next_state, reward, terminated = self.get_transition(state, action)
                value = reward + (0 if terminated else self.gamma * self.V[next_state])
                
                if value > best_value:
                    best_value = value
                    best_action = action
            
            self.policy[state] = best_action
    
    def select_action(self, state):
        """Sélectionne l'action selon la politique"""
        return self.policy.get(tuple(state), 0)


def get_random_goal_position(env):
    """Génère une position aléatoire pour le goal"""
    valid_positions = []
    for row in range(env.height):
        for col in range(env.width):
            pos = (row, col)
            if (pos not in env.obstacles and 
                pos != env.start_pos and
                abs(pos[0] - env.start_pos[0]) + abs(pos[1] - env.start_pos[1]) > 5):
                valid_positions.append(pos)
    
    return valid_positions[np.random.randint(len(valid_positions))]


def main():
    print("=" * 70)
    print(" " * 5 + "PROGRAMME 3 - GOAL SE DÉPLACE ENTRE ÉPISODES")
    print("=" * 70)
    print("\nLe goal change de position à chaque épisode")
    print("L'agent réapprend la politique optimale à chaque fois\n")
    
    # Configuration initiale
    config = GridWorldConfig(
        grid_size=(10, 10),
        agent_start=(0, 0),
        goals=[(9, 9)],  # Position initiale
        obstacles=[(4, 4), (4, 5), (5, 4), (5, 5)],
        max_steps=100
    )
    
    # Créer l'environnement
    env = GridWorldGymnasium(config=config, render_mode="human")
    
    print(f"Grid: {env.height}x{env.width}")
    print(f"Start: {env.start_pos}")
    print(f"Obstacles: {len(env.obstacles)}")
    
    # Créer l'agent
    agent = ValueIterationAgent(env, gamma=0.9)
    
    # Paramètres
    n_episodes = int(input("\nNombre d'épisodes (default=5): ").strip() or "5")
    delay = float(input("Vitesse animation (0.1=rapide, 0.5=lent, default=0.2): ").strip() or "0.2")
    
    print("\n" + "=" * 70)
    print("ÉPISODES AVEC GOAL MOBILE")
    print("=" * 70)
    
    stats = {'rewards': [], 'steps': [], 'iterations': []}
    
    for episode in range(n_episodes):
        print(f"\n{'='*70}")
        print(f"ÉPISODE {episode + 1}/{n_episodes}")
        print(f"{'='*70}")
        
        # Changer la position du goal
        new_goal = get_random_goal_position(env)
        env.goals = {new_goal}
        print(f"📍 Nouveau GOAL: {new_goal}")
        
        # Réentraîner l'agent avec le nouveau goal
        n_iterations = agent.value_iteration(verbose=True)
        agent.extract_policy()
        
        # Configurer l'affichage
        env.set_render_params(value_states=agent.V, policy=agent.policy)
        
        # Tester la politique
        observation, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        print(f"🎬 Exécution de l'épisode...")
        env.render()
        plt.pause(1.0)
        
        while not (terminated or truncated):
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if step % 5 == 0 or terminated:
                print(f"Step {step}: {tuple(observation)}")
            
            env.render()
            plt.pause(delay)
            
            if terminated:
                print(f"🎉 GOAL ATTEINT en {step} steps!")
        
        stats['rewards'].append(total_reward)
        stats['steps'].append(step)
        stats['iterations'].append(n_iterations)
        
        plt.pause(1.5)
    
    # Statistiques globales
    print("\n" + "=" * 70)
    print("STATISTIQUES GLOBALES")
    print("=" * 70)
    print(f"Épisodes: {n_episodes}")
    print(f"Steps moyens: {np.mean(stats['steps']):.1f}")
    print(f"Récompense moyenne: {np.mean(stats['rewards']):.3f}")
    print(f"Itérations moyennes (convergence): {np.mean(stats['iterations']):.1f}")
    print(f"Meilleur épisode: {min(stats['steps'])} steps")
    
    env.close()
    print("\n" + "=" * 70)
    print("Programme terminé!")
    print("=" * 70)


if __name__ == "__main__":
    main()
