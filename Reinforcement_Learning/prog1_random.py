"""
PROGRAMME 1: Agent Random
L'agent se déplace aléatoirement jusqu'à trouver le goal
"""

import numpy as np
import matplotlib.pyplot as plt
from gym_environment import GridWorldGymnasium
from config import GridWorldConfig


def main():
    print("=" * 70)
    print(" " * 15 + "PROGRAMME 1 - AGENT RANDOM")
    print("=" * 70)
    print("\nL'agent se déplace aléatoirement jusqu'à trouver le goal\n")
    
    # Configuration
    config = GridWorldConfig(
        grid_size=(8, 8),
        agent_start=(0, 0),
        goals=[(7, 7)],
        obstacles=[(3, 3), (3, 4), (4, 3)],
        max_steps=200
    )
    
    # Créer l'environnement
    env = GridWorldGymnasium(config=config, render_mode="human")
    
    print(f"Grid: {env.height}x{env.width}")
    print(f"Start: {env.start_pos}")
    print(f"Goal: {list(env.goals)[0]}")
    print(f"Obstacles: {len(env.obstacles)}")
    print(f"Max steps: {config.max_steps}")
    
    # Paramètres
    n_episodes = int(input("\nNombre d'épisodes (default=5): ").strip() or "5")
    delay = float(input("Vitesse animation (0.1=rapide, 0.5=lent, default=0.2): ").strip() or "0.2")
    
    print("\n" + "=" * 70)
    print("EXPLORATION ALÉATOIRE")
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
            # Action aléatoire
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Afficher tous les 10 steps
            if step % 10 == 0 or terminated:
                print(f"Step {step}: Position={tuple(observation)}, Reward Total={total_reward:.2f}")
            
            env.render()
            plt.pause(delay)
            
            if terminated:
                print(f"\n🎉 GOAL TROUVÉ en {step} steps!")
            elif truncated:
                print(f"\n⏱️ Timeout après {step} steps")
        
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
    print(f"Meilleur épisode: {min(stats['steps'])} steps")
    print(f"Pire épisode: {max(stats['steps'])} steps")
    
    env.close()
    print("\n" + "=" * 70)
    print("Programme terminé!")
    print("=" * 70)


if __name__ == "__main__":
    main()
