import os
import sys

# Ensure env module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.rerouting_env import ReroutingEnv

def test_environment_initialization():
    cfg_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "grid.sumocfg")
    print(f"Testing environment with config: {cfg_file}")
    
    env = None
    try:
        env = ReroutingEnv(sumocfg_file=cfg_file, use_gui=False)
        print("Observation Space:", env.observation_space)
        print("Action Space:", env.action_space)
        
        print("Resetting environment...")
        obs, info = env.reset()
        print("Initial Observation:", obs)
        
        # Run a brief simulation loop
        print("Running 5 steps with random actions...")
        for i in range(5):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            print(f"Step {i+1}: Action={action}, Reward={reward:.2f}, Terminated={terminated}, Truncated={truncated}, Obs={obs}")
            if terminated or truncated:
                break
                
        print("Environment test completed successfully!")
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        if env is not None:
            env.close()

if __name__ == "__main__":
    test_environment_initialization()
