"""
PROGRAMME 4: Value Iteration avec Goal qui se déplace PENDANT l'épisode
Le goal se déplace à chaque N steps pendant l'épisode en cours
L'agent doit s'adapter en temps réel
"""

import numpy as np
import matplotlib.pyplot as plt
from gym_environment import GridWorldGymnasium
from config import GridWorldConfig


class AdaptiveValueIterationAgent:
    """Agent qui s'adapte au goal mobile"""
    
    def __init__(self, env, gamma=0.9, theta=1e-4):
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
    
    def quick_value_iteration(self, max_iterations=50):
        """Value Iteration rapide pour adaptation en temps réel"""
        # Réinitialiser les valeurs
        self.V = {state: 0.0 for state in self.states}
        
        for iteration in range(max_iterations):
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
            
            if delta < self.theta:
                return iteration + 1
        
        return max_iterations
    
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


def get_random_goal_position(env, current_goal=None):
    """Génère une nouvelle position pour le goal"""
    valid_positions = []
    for row in range(env.height):
        for col in range(env.width):
            pos = (row, col)
            if (pos not in env.obstacles and 
                pos != env.start_pos and
                pos != current_goal):
                valid_positions.append(pos)
    
    return valid_positions[np.random.randint(len(valid_positions))]


def main():
    print("=" * 70)
    print(" " * 5 + "PROGRAMME 4 - GOAL SE DÉPLACE PENDANT L'ÉPISODE")
    print("=" * 70)
    print("\nLe goal se déplace toutes les N steps PENDANT l'épisode")
    print("L'agent doit s'adapter en temps réel!\n")
    
    # Configuration
    config = GridWorldConfig(
        grid_size=(10, 10),
        agent_start=(0, 0),
        goals=[(9, 9)],
        obstacles=[(4, 4), (4, 5), (5, 4)],
        max_steps=150  # Plus de steps car le goal bouge
    )
    
    # Créer l'environnement
    env = GridWorldGymnasium(config=config, render_mode="human")
    
    print(f"Grid: {env.height}x{env.width}")
    print(f"Start: {env.start_pos}")
    print(f"Obstacles: {len(env.obstacles)}")
    
    # Créer l'agent
    agent = AdaptiveValueIterationAgent(env, gamma=0.9, theta=1e-4)
    
    # Paramètres
    n_episodes = int(input("\nNombre d'épisodes (default=3): ").strip() or "3")
    steps_before_move = int(input("Goal bouge tous les N steps (default=15): ").strip() or "15")
    delay = float(input("Vitesse animation (0.05=rapide, 0.3=lent, default=0.15): ").strip() or "0.15")
    
    print("\n" + "=" * 70)
    print("ÉPISODES AVEC GOAL DYNAMIQUE")
    print("=" * 70)
    
    stats = {'rewards': [], 'steps': [], 'goal_changes': [], 'adaptations': []}
    
    for episode in range(n_episodes):
        print(f"\n{'='*70}")
        print(f"ÉPISODE {episode + 1}/{n_episodes}")
        print(f"{'='*70}")
        
        # Position initiale du goal
        current_goal = get_random_goal_position(env)
        env.goals = {current_goal}
        print(f"📍 Goal initial: {current_goal}")
        
        # Entraînement initial
        print(f"🔄 Apprentissage initial...")
        agent.quick_value_iteration(max_iterations=50)
        agent.extract_policy()
        env.set_render_params(value_states=agent.V, policy=agent.policy)
        
        # Démarrer l'épisode
        observation, info = env.reset()
        total_reward = 0
        terminated = False
        truncated = False
        step = 0
        goal_changes = 0
        adaptations = 0
        
        print(f"\n🎬 Démarrage de l'épisode...")
        env.render()
        plt.pause(1.0)
        
        while not (terminated or truncated):
            # Déplacer le goal tous les N steps
            if step > 0 and step % steps_before_move == 0:
                old_goal = current_goal
                current_goal = get_random_goal_position(env, current_goal)
                env.goals = {current_goal}
                goal_changes += 1
                
                print(f"\n🔄 GOAL DÉPLACÉ! {old_goal} → {current_goal}")
                print(f"   Réapprentissage en cours...")
                
                # Réapprendre rapidement
                n_iter = agent.quick_value_iteration(max_iterations=50)
                agent.extract_policy()
                env.set_render_params(value_states=agent.V, policy=agent.policy)
                adaptations += 1
                
                print(f"   ✅ Adapté en {n_iter} itérations\n")
            
            # Sélectionner et exécuter l'action
            action = agent.select_action(observation)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            if step % 5 == 0 or terminated:
                distance = abs(observation[0] - current_goal[0]) + abs(observation[1] - current_goal[1])
                print(f"Step {step}: {tuple(observation)} | Distance au goal: {distance}")
            
            env.render()
            plt.pause(delay)
            
            if terminated:
                print(f"\n🎉 GOAL ATTEINT en {step} steps!")
                print(f"   Le goal a bougé {goal_changes} fois")
        
        if truncated:
            print(f"\n⏱️ Timeout après {step} steps (goal a bougé {goal_changes} fois)")
        
        stats['rewards'].append(total_reward)
        stats['steps'].append(step)
        stats['goal_changes'].append(goal_changes)
        stats['adaptations'].append(adaptations)
        
        plt.pause(2.0)
    
    # Statistiques globales
    print("\n" + "=" * 70)
    print("STATISTIQUES GLOBALES")
    print("=" * 70)
    print(f"Épisodes: {n_episodes}")
    print(f"Steps moyens: {np.mean(stats['steps']):.1f}")
    print(f"Récompense moyenne: {np.mean(stats['rewards']):.3f}")
    print(f"Changements de goal moyens: {np.mean(stats['goal_changes']):.1f}")
    print(f"Adaptations moyennes: {np.mean(stats['adaptations']):.1f}")
    successes = sum([r > 0 for r in stats['rewards']])
    print(f"Succès: {successes}/{n_episodes}")
    
    env.close()
    print("\n" + "=" * 70)
    print("Programme terminé!")
    print("=" * 70)


if __name__ == "__main__":
    main()
