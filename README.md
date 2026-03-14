# Dynamic Rerouting Agent (Model 3)

This module implements a deep Reinforcement Learning (RL) agent to dynamically optimize vehicle routing in traffic environments. It uses **Stable-Baselines3 (DQN)** for decision-making and interacts directly with the **SUMO Traffic Simulator** via the **TraCI** API.

This system is designed to act as the master controller in a Multi-Model Architecture, digesting outputs from Model 1 (Traffic Prediction) and Model 2 (Congestion Detection) to mitigate routing bottlenecks in real-time.

## Features
- **SUMO TraCI Integration**: Directly interacts with OpenStreetMap networks inside the SUMO graphical simulation.
- **Deep Q-Network (DQN) PyTorch Backend**: Continuously learns optimal routing paths based on state vectors.
- **Multi-Model JSON Standardized API**: Specifically designed for strict JSON compatibility with prediction and congestion arrays.
- **Strict Safety Boundaries**: Enforces hard anti-oscillation limits (`max_reroutes = 2`, `timeout = 60s`) and proxy locks (<300m restrictions).

---

## 🚀 Quickstart & Demo

If you want to watch the PyTorch DQN agent visually route vehicles inside the simulation right now:

```bash
# 1. Activate your virtual environment
.\venv\Scripts\activate

# 2. Run the Evaluation Script (Pops open the native SUMO GUI!)
python -u agent/evaluate_bangalore.py
```
*The simulation will run a central bounding box of Bangalore pulled from OpenStreetMap, and end-of-episode telemetry (`avg_travel_time`, `reroutes`) will output to your terminal.*

---

## 🔌 API Integration (Production Use)

When you are ready to integrate this project with **Model 1** and **Model 2**, use the formalized Routing Service API (`api/routing_service.py`). 

You do **not** run the SUMO GUI. You simply pass generic dictionaries, and the RL service will instantly infer the decision algorithm and return the optimal action.

### Example Integration Script:

```python
from api.routing_service import RoutingServiceAPI

# 1. Start the Routing Service (loads the best PyTorch DQN weights)
api = RoutingServiceAPI(model_path="models/best_model", env_net=None)

# 2. Pass YOUR REAL INPUTS (JSON format from Model 1 and Model 2)
vehicle_state = {
  "vehicle_id": "ego_0",
  "position": "edge_45",
  "current_speed": 15.5,
  "remaining_distance": 1200.0,
}

model1_prediction = {
  "predicted_density": 0.85, # From Model 1
}

model2_congestion = {
  "congested_edges": ["edge_46", "edge_47"], # From Model 2
  "queue_length": 25.0,
}

# 3. GET THE AGENT'S DECISION based on your incoming JSON
decision = api.predict_best_route(vehicle_state, model1_prediction, model2_congestion)

print(decision)
# Output -> {"vehicle_id": "ego_0", "action": "reroute", "new_route": ["edge_45", "edge_52"]}
```

### Run the API Test Mock Script
You can verify the exact code above runs smoothly on your machine utilizing the mock script:

```bash
python -u tests/test_api_integration.py
```

---

## Architecture details

### The State Vector (`Box(6,)`)
The internal Neural Network has exactly 6 input nodes mapping to dynamic traffic telemetry:
1. `vehicle_speed` (Local proxy)
2. `remaining_distance` (Local proxy)
3. `local_density` (Local proxy)
4. `predicted_density_ahead` (Model 1 metric)
5. `target_queue_length` (Model 2 metric)
6. `num_alternative_paths_available` 

### The Action Space (`Discrete(3)`)
The agent enforces one of three outputs every inference tick (`15s`):
- `0` : Stay on current route
- `1` : Reroute to Path Alternative A
- `2` : Reroute to Path Alternative B
