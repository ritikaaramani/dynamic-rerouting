import os
import sys

# Ensure env module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.rerouting_env import ReroutingEnv

from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import EvalCallback

def train_agent():
    cfg_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "grid.sumocfg")
    
    print("Initializing environment...")
    # Training environment shouldn't use GUI for speed
    env = ReroutingEnv(sumocfg_file=cfg_file, use_gui=False)
    
    # Eval environment
    eval_env = ReroutingEnv(sumocfg_file=cfg_file, use_gui=False)

    print("Initializing DQN Agent...")
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=1e-3,
        buffer_size=10000,
        learning_starts=1000,
        batch_size=64,
        gamma=0.99,
        target_update_interval=500,
        exploration_fraction=0.2,
        exploration_final_eps=0.05,
        verbose=1,
        tensorboard_log="./dqn_rerouting_tensorboard/"
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./models/',
        log_path='./logs/',
        eval_freq=1000,
        deterministic=True,
        render=False
    )

    print("Beginning Training...")
    # Train for 10,000 steps
    try:
        model.learn(total_timesteps=10000, callback=eval_callback, log_interval=4)
        print("Training complete!")
        
        # Save final model
        model.save("models/dqn_rerouting_final")
        
    except Exception as e:
        print(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        env.close()
        eval_env.close()

if __name__ == "__main__":
    train_agent()
