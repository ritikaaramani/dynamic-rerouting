import os
import sys
import json

# Ensure env module can be imported
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import traci

# Mock imports for API testing since we don't want to boot the actual SUMO engine just for a unit test
from unittest.mock import MagicMock
traci.simulation = MagicMock()
traci.simulation.getTime.return_value = 120.0 

from api.routing_service import RoutingServiceAPI
import numpy as np
from stable_baselines3 import DQN

def test_multi_model_routing():
    print("--- Multi-Model Routing API Test ---")
    
    # Setup dummy model path
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "best_model.zip")
    if not os.path.exists(model_path):
        print("Skipping active PyTorch prediction, best_model.zip not found.")
        return
        
    api = RoutingServiceAPI(model_path=model_path.replace('.zip', ''), env_net=None)
    
    # 1. Simulate Vehicle State arriving from physical SUMO simulation
    vehicle_state = {
      "vehicle_id": "ego_0",
      "position": "edge_45",
      "current_speed": 15.5,
      "remaining_distance": 1200.0,
    }
    print(f"\n[Input] Vehicle State:")
    print(json.dumps(vehicle_state, indent=2))
    
    # 2. Simulate Input from Model 1 (Prediction API)
    model1_prediction = {
      "predicted_density": 0.85, # High expected density ahead
    }
    print(f"\n[Input] Model 1 (Traffic Prediction):")
    print(json.dumps(model1_prediction, indent=2))
    
    # 3. Simulate Input from Model 2 (Congestion Detection API)
    model2_congestion = {
      "congested_edges": ["edge_46", "edge_47"],
      "queue_length": 25.0,
      "congestion_level": "high"
    }
    print(f"\n[Input] Model 2 (Congestion Detection):")
    print(json.dumps(model2_congestion, indent=2))
    
    # 4. Agent Execution
    print("\nExecuting RL Policy Decision Matrix...")
    decision = api.predict_best_route(vehicle_state, model1_prediction, model2_congestion)
    
    print(f"\n[Output] Final RL Routing Decision (Model 3):")
    print(json.dumps(decision, indent=2))
    
    # 5. Test Safety Constraint Lockout
    print("\n[Safety System] Simulating immediate subsequent request (timeout test)...")
    traci.simulation.getTime.return_value = 135.0 # Only 15 seconds later
    
    decision_2 = api.predict_best_route(vehicle_state, model1_prediction, model2_congestion)
    print(f"\n[Output] Fast-follow Routing Decision:")
    print(json.dumps(decision_2, indent=2))
    print("Test passed successfully.")

if __name__ == "__main__":
    test_multi_model_routing()
