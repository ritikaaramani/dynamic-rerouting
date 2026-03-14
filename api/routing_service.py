import os
import sys

# Ensure SUMO_HOME is set and tools/bin are on PATH
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    bin_path = os.path.join(os.environ['SUMO_HOME'], 'bin')
    sys.path.append(tools)
    if bin_path not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + bin_path

import traci
import numpy as np
from stable_baselines3 import DQN

class RoutingServiceAPI:
    """
    Standardized API Layer to interface the RL Model 3 Agent with 
    Model 1 (Prediction) and Model 2 (Congestion).
    """
    def __init__(self, model_path, env_net):
        """
        Initializes the service by loading the trained PyTorch Policy Network
        and storing a reference to the active road graph.
        """
        self.model = DQN.load(model_path)
        self.net = env_net
        
        # History dict to track constraints (timeout, max changes)
        self.vehicle_history = {} 

    def get_vehicle_state(self, vehicle_id):
        """
        Pulls real-time localized attributes from SUMO for a target vehicle via TraCI.
        Matches the required input state JSON format.
        """
        try:
            curr_edge = traci.vehicle.getRoadID(vehicle_id)
            if curr_edge.startswith(':'):
                curr_edge = traci.vehicle.getRoute(vehicle_id)[traci.vehicle.getRouteIndex(vehicle_id)]
                
            speed = traci.vehicle.getSpeed(vehicle_id)
            route = traci.vehicle.getRoute(vehicle_id)
            dest_edge = route[-1]
            rem_dist = traci.vehicle.getDrivingDistance(vehicle_id, dest_edge, 0.0)

            return {
                "vehicle_id": vehicle_id,
                "position": curr_edge,
                "current_speed": speed,
                "remaining_distance": max(rem_dist, 0.0)
            }
        except traci.exceptions.TraCIException:
            return None # Vehicle finished or not found

    def generate_alternative_routes(self, vehicle_id, current_edge, dest_edge, k=3):
        """
        Calculates k distinct routing alternatives via the SUMO routing engine.
        For simplicity, we currently pull standard fast alternatives via TraCI native.
        """
        # Traci currently supports optimal rerouting directly via rerouteTraveltime
        # A more complex engine would compute explicitly K distinct paths. For API spec mapping:
        # Assuming we return dummy alternative edges here representing the 'Alternative Action Space'
        return [f"alt_path_1_{current_edge}", f"alt_path_2_{current_edge}"]

    def predict_best_route(self, vehicle_state, model1_prediction, model2_congestion, criticality_level="Medium"):
        """
        The Main Integration Hook. Takes the Base State + M1 + M2 and constructs the 
        Box(6,) State Vector, queries the RL agent, and returns the formal routing Decision JSON.
        
        Phase 2: Added high-criticality bypass for safety constraints.
        """
        vid = vehicle_state["vehicle_id"]
        
        # Parse M1 Features
        pred_density = model1_prediction.get("predicted_density", 0.0)
        
        # Parse M2 Features
        # Approximating queue length from congestion array
        congested_edges = model2_congestion.get("congested_edges", [])
        queue_len = len(congested_edges) * 10.0 # dummy scaler based on array len
        
        route_alts = self.generate_alternative_routes(vid, vehicle_state["position"], "dest_edge", k=2)

        # 1. State Builder: Flatten into expected RL observation array
        # [speed, rem_distance, density, pred_density, queue_len, num_alts]
        state_vector = np.array([
            min(vehicle_state["current_speed"] / 30.0, 1.0),
            min(vehicle_state["remaining_distance"] / 5000.0, 1.0),
            0.5, # internal traffic localized density placeholder
            min(pred_density, 1.0),
            min(queue_len / 50.0, 1.0), # normalized
            len(route_alts)
        ], dtype=np.float32)

        # 2. Safety Rules Constraints Evaluation
        history = self.vehicle_history.get(vid, {"reroutes": 0, "last_reroute_time": -999})
        current_time = traci.simulation.getTime()
        
        # Phase 2: High Criticality bypasses max_reroute and timeout constraints
        is_urgent = (criticality_level == "High")
        
        if not is_urgent:
            if history["reroutes"] >= 2:
                return {"vehicle_id": vid, "action": "stay", "reason": "max_reroutes_reached"}
                
            if current_time - history["last_reroute_time"] < 60.0:
                return {"vehicle_id": vid, "action": "stay", "reason": "timeout_lock"}
            
        if vehicle_state["remaining_distance"] < 300.0:
            return {"vehicle_id": vid, "action": "stay", "reason": "proximity_constraint"}

        # 3. RL Agent Predicts Action via Policy
        action, _ = self.model.predict(state_vector, deterministic=True)
        
        # 4. Map Action to JSON Output
        if action == 0:
            decision = {"vehicle_id": vid, "action": "stay"}
        else:
            decision = {
                "vehicle_id": vid, 
                "action": "reroute", 
                "new_route": route_alts # Mapping dummy action 1 or 2 mapping
            }
            # Commit routing to history immediately
            history["reroutes"] += 1
            history["last_reroute_time"] = current_time
            self.vehicle_history[vid] = history
            
        # Phase 2: Log to Dispatch Monitoring Stream
        self._log_dispatch_event(vid, decision, vehicle_state, criticality_level)
            
        return decision

    def _log_dispatch_event(self, vehicle_id, decision, state, criticality):
        """
        Simulates a Live Dispatch Monitoring Webhook/Stream.
        """
        log_entry = {
            "timestamp": traci.simulation.getTime(),
            "vehicle": vehicle_id,
            "criticality": criticality,
            "location": state["position"],
            "speed": state["current_speed"],
            "decision": decision["action"],
            "eta_impact": "calculating..."
        }
        # In production, this would POST to a URL or a WebSocket
        print(f"[DISPATCH LOG] {log_entry}")

    def inject_blockage(self, edge_id):
        """
        Phase 2: Forces an immediate road blockage in SUMO to trigger RL rerouting.
        """
        try:
            # Setting max speed to near-zero effectively blocks the edge
            traci.edge.setMaxSpeed(edge_id, 0.1)
            # Notify routing engine that edge travel time is now infinite
            traci.edge.setEffort(edge_id, 999999)
            print(f"Road Blockage injected on {edge_id}")
            return True
        except Exception as e:
            print(f"Failed to inject blockage on {edge_id}: {e}")
            return False

    def apply_reroute(self, vehicle_id, new_route_edges):
        """
        Fires TraCI API to force the active vehicle onto the specified edge sequence list.
        """
        try:
            # Native TraCI applies routing via ID list
            # In a full multi-model setup, new_route_edges would exactly match physical edges
            # traci.vehicle.setRoute(vehicle_id, new_route_edges)
            traci.vehicle.rerouteTraveltime(vehicle_id) # Using dynamic native as fallback
            return True
        except Exception as e:
            print(f"Failed to apply reroute for {vehicle_id}: {e}")
            return False
