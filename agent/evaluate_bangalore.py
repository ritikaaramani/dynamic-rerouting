import os
import sys

# Ensure env module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.rerouting_env import ReroutingEnv

from stable_baselines3 import DQN

def evaluate_on_bangalore():
    cfg_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bangalore.sumocfg")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Trained model not found at {model_path}.zip")
        return

    print("Loading test environment (Bangalore Real Map)...")
    env = ReroutingEnv(sumocfg_file=cfg_file, use_gui=True)
    
    print("Loading DQN Agent...")
    model = DQN.load(model_path)
    
    print("Evaluating over 5 episodes on unseen Bangalore map...")
    for ep in range(5):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            done = terminated or truncated
            
        print(f"Episode {ep + 1}/{5} finished after {steps} steps with total reward: {total_reward:.2f}")
        if 'avg_travel_time' in info:
            print(f"Performance -> Avg Travel Time: {info['avg_travel_time']}s | Avg Waiting Time: {info['avg_waiting_time']}s | Reroutes: {info['reroutes']}")

    print("Bangalore Evaluation complete!")
    env.close()

if __name__ == "__main__":
    evaluate_on_bangalore()
