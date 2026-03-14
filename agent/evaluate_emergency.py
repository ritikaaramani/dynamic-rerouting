import os
import sys
import time

# Ensure env module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from env.rerouting_env import ReroutingEnv
from api.routing_service import RoutingServiceAPI
from stable_baselines3 import DQN

def demo_emergency_response():
    cfg_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "bangalore.sumocfg")
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model")
    
    print("--- Phase 2: Emergency Response Simulation ---")
    print("Initializing Environment: Weather=Rain, Criticality=High")
    
    # Initialize environment with Phase 2 options
    env = ReroutingEnv(sumocfg_file=cfg_file, use_gui=True)
    obs, info = env.reset(options={"weather": "Rain", "criticality": "High"})
    
    # Initialize API for advanced decisions
    api = RoutingServiceAPI(model_path=model_path, env_net=None)
    
    done = False
    step_count = 0
    
    print("\nStarting Simulation Loop...")
    while not done:
        # 1. Fetch current vehicle state from SUMO
        vehicle_state = api.get_vehicle_state("ego_0")
        if not vehicle_state:
            break
            
        # 2. Simulate external Model 1 and Model 2 inputs
        m1_pred = {"predicted_density": 0.6}
        m2_cong = {"congested_edges": ["edge_1", "edge_2"]}
        
        # 3. Decision via API (with Criticality context)
        decision = api.predict_best_route(
            vehicle_state, 
            m1_pred, 
            m2_cong, 
            criticality_level="High"
        )
        
        # 4. Trigger Dynamic Blockage mid-way
        if step_count == 10:
            print("\n!!! DISRUPTION DETECTED: Injecting road blockage on downstream edge !!!")
            # In a real scenario, this edge would be ahead of the vehicle
            # We'll just pick a random edge from the environment list for the demo
            api.inject_blockage(env.edges[20])
            
        # Execute Action in Env
        action_map = {"stay": 0, "reroute": 1}
        action = action_map.get(decision["action"], 0)
        
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1
        
        # Slow down slightly for visual clarity in GUI
        time.sleep(0.1)

    print("\n--- Emergency Mission Complete ---")
    if "avg_travel_time" in info:
        print(f"Final Report:")
        print(f">> Total Travel Time: {info['avg_travel_time']}s (Rain Conditions)")
        print(f">> Emergency Reroutes: {info['reroutes']}")
    
    env.close()

if __name__ == "__main__":
    demo_emergency_response()
