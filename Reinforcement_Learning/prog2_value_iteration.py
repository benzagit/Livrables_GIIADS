"""
PROGRAMME 2: Agent avec Value Iteration et Policy
L'agent apprend la politique optimale avec Value Iteration
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
    
    def value_iteration(self):
        """Algorithme Value Iteration"""
        print("\n🔄 Entraînement Value Iteration...")
        
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
            
            if (iteration + 1) % 10 == 0:
                print(f"  Iteration {iteration + 1}: Delta={delta:.6f}")
            
            if delta < self.theta:
                print(f"✅ Convergence en {iteration + 1} itérations!\n")
                return iteration + 1
        
        return 1000
    
    def extract_policy(self):
        """Extrait la politique optimale"""
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


def main():
    print("=" * 70)
    print(" " * 10 + "PROGRAMME 2 - VALUE ITERATION + POLICY")
    print("=" * 70)
    print("\nL'agent apprend la politique optimale avec Value Iteration\n")
    
    # Configuration
    config = GridWorldConfig(
        grid_size=(8, 8),
        agent_start=(0, 0),
        goals=[(7, 7)],
        obstacles=[(3, 3), (3, 4), (4, 3)],
        max_steps=100
    )
    
    # Créer l'environnement
    env = GridWorldGymnasium(config=config, render_mode="human")
    
    print(f"Grid: {env.height}x{env.width}")
    print(f"Start: {env.start_pos}")
    print(f"Goal: {list(env.goals)[0]}")
    print(f"Obstacles: {len(env.obstacles)}")
    
    # Créer et entraîner l'agent
    agent = ValueIterationAgent(env, gamma=0.9)
    n_iterations = agent.value_iteration()
    agent.extract_policy()
    
    # Configurer l'affichage avec value states et policy
    env.set_render_params(value_states=agent.V, policy=agent.policy)
    
    print("📊 Affichage de la grille avec Value States et Policy...")
    env.reset()
    env.render()
    plt.pause(2.0)
    
    # Tester la politique
    n_episodes = int(input("\nNombre d'épisodes de test (default=3): ").strip() or "3")
    delay = float(input("Vitesse animation (0.1=rapide, 0.5=lent, default=0.3): ").strip() or "0.3")
    
    print("\n" + "=" * 70)
    print("TEST DE LA POLITIQUE OPTIMALE")
    print("=" * 70)
    
    stats = {'rewards': [], 'steps': [], 'successes': []}
    
    for episode in range(n_episodes):
        print(f"\n{'='*70}")
        print(f"ÉPISODE {episode + 1}/{n_episodes}")
        print(f"{'='*70}")
        
        observation, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        step = 0
        
        env.render()
        plt.pause(0.5)
        
        while not (terminated or truncated):
            # Action selon la politique
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            print(f"Step {step}: Action={env.ACTION_NAMES[action]:6} -> {tuple(observation)}")
            
            env.render()
            plt.pause(delay)
            
            if terminated:
                print(f"\n🎉 GOAL ATTEINT en {step} steps!")
        
        stats['rewards'].append(total_reward)
        stats['steps'].append(step)
        stats['successes'].append(info.get('success', False))
        
        plt.pause(1.0)
    
    # Statistiques
    print("\n" + "=" * 70)
    print("STATISTIQUES")
    print("=" * 70)
    print(f"Taux de succès: {np.mean(stats['successes']):.1%}")
    print(f"Steps moyens: {np.mean(stats['steps']):.1f}")
    print(f"Récompense moyenne: {np.mean(stats['rewards']):.3f}")
    print(f"Convergence: {n_iterations} itérations")
    
    env.close()
    print("\n" + "=" * 70)
    print("Programme terminé!")
    print("=" * 70)


if __name__ == "__main__":
    main()
