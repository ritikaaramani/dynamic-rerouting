import os
import sys
import gymnasium as gym
from gymnasium import spaces
import numpy as np

# Ensure SUMO_HOME is set and tools/bin are on PATH
if 'SUMO_HOME' not in os.environ:
    os.environ['SUMO_HOME'] = r"C:\Program Files (x86)\Eclipse\Sumo"

if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    bin_path = os.path.join(os.environ['SUMO_HOME'], 'bin')
    sys.path.append(tools)
    if bin_path not in os.environ['PATH']:
        os.environ['PATH'] += os.pathsep + bin_path
else:
    sys.exit("Please declare environment variable 'SUMO_HOME'")

import traci
import sumolib

class ReroutingEnv(gym.Env):
    """
    Custom Environment that follows gym interface for a Dynamic Rerouting Agent.
    """
    metadata = {'render.modes': ['human', 'none']}

    def __init__(self, sumocfg_file, use_gui=False, ego_vehicle_id="ego_0"):
        super(ReroutingEnv, self).__init__()
        self.sumocfg_file = sumocfg_file
        self.use_gui = use_gui
        self.ego_vehicle_id = ego_vehicle_id
        
        # Action space: 0=Stay, 1=Switch (Alt A), 2=Switch (Alt B)
        self.action_space = spaces.Discrete(3)
        
        # State space: [current_edge_idx, distance_to_dest, local_density,
        #               predicted_congestion (M1), queue_length (M2), num_alts]
        # Using normalized values roughly bounded [-1, 1] or [0, 1] for neural net friendliness.
        self.observation_space = spaces.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        
        self.sim_step = 0
        self.max_steps = 1000
        self.sumo_cmd = ["sumo-gui" if use_gui else "sumo", "-c", self.sumocfg_file, "--no-step-log", "true", "--waiting-time-memory", "10000"]
        self.traci_connection = None
        
        # Load network to query edges
        cfg_dir = os.path.dirname(self.sumocfg_file)
        net_filename = os.path.basename(self.sumocfg_file).replace('.sumocfg', '.net.xml')
        self.net = sumolib.net.readNet(os.path.join(cfg_dir, net_filename), withPrograms=True)
        self.edges = [e.getID() for e in self.net.getEdges()]
        
        # Tracking states
        self.last_action = 0
        self.destination_edge = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Metrics Tracking
        self.total_travel_time = 0.0
        self.total_waiting_time = 0.0
        self.total_reroutes = 0
        
        # Close any existing connection
        try:
            traci.close()
        except traci.exceptions.FatalTraCIError:
            pass

        # Start TraCI
        traci.start(self.sumo_cmd)
        self.sim_step = 0
        
        # We need to ensure our ego vehicle is in the simulation
        # For simplicity, we inject it manually at the start of the episode
        start_edge = self.edges[0]
        self.destination_edge = self.edges[-1] # Target last edge
        
        route_id = "ego_route"
        if route_id not in traci.route.getIDList():
            traci.route.add(route_id, [start_edge, self.destination_edge])
            
        if self.ego_vehicle_id not in traci.vehicle.getIDList():
            traci.vehicle.add(self.ego_vehicle_id, route_id, typeID="DEFAULT_VEHTYPE")
            traci.vehicle.setColor(self.ego_vehicle_id, (255, 0, 0, 255)) # Make ego vehicle red
            
        # Step until ego enters the network
        while self.ego_vehicle_id not in traci.vehicle.getIDList():
            traci.simulationStep()
            self.sim_step += 1
            if self.sim_step > 100:
                raise RuntimeError("Ego vehicle did not enter the network.")

        # Phase 2: Emergency Response State
        self.criticality_level = options.get("criticality", "High") if options else "High"
        self.weather = options.get("weather", "Clear") if options else "Clear"
        
        # Apply Weather scaling to the network
        if self.weather != "Clear":
            # Scale down speeds for all edges to simulate rain/snow
            scale = 0.7 if self.weather == "Rain" else 0.4
            for edge_id in self.edges:
                orig_speed = traci.edge.getMaxSpeed(edge_id)
                traci.edge.setMaxSpeed(edge_id, orig_speed * scale)
        
        return self._get_obs(), {}

    def _get_obs(self):
        try:
            if self.ego_vehicle_id not in traci.vehicle.getIDList():
                return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
                
            curr_edge = traci.vehicle.getRoadID(self.ego_vehicle_id)
            if curr_edge.startswith(':'): # Internal intersection edge
                curr_edge = traci.vehicle.getRoute(self.ego_vehicle_id)[traci.vehicle.getRouteIndex(self.ego_vehicle_id)]
                
            edge_idx = self.edges.index(curr_edge) if curr_edge in self.edges else 0
            
            # 1. Normalized edge idx
            norm_edge = edge_idx / max(1, len(self.edges))
            
            # 2. Distance to destination
            dist = traci.vehicle.getDrivingDistance(self.ego_vehicle_id, self.destination_edge, 0.0)
            if dist < 0: dist = 0.0 # Error fallback
            max_dist = 2000.0 # Assuming 2km max for grid
            norm_dist = min(dist / max_dist, 1.0)
            
            # 3. Predicted traffic density (simple approximation: number of vehicles on current edge)
            veh_on_edge = traci.edge.getLastStepVehicleNumber(curr_edge)
            max_veh = 50.0 
            norm_density = min(veh_on_edge / max_veh, 1.0)
            
            # 4. Simulated Model 1 Prediction: Congestion probability
            # (In production, passed directly via API)
            sim_pred_congestion = min((veh_on_edge * 1.5) / max_veh, 1.0)
            
            # 5. Simulated Model 2 Congestion: Target edge queue length
            sim_queue_length = min(veh_on_edge / 10.0, 1.0)
            
            # 6. Available route alternatives (always 2 for this DQN env action space mapping)
            num_alts = 2.0 / 5.0 # normalized
            
            return np.array([norm_edge, norm_dist, norm_density, sim_pred_congestion, sim_queue_length, num_alts], dtype=np.float32)
        except traci.exceptions.TraCIException:
            # Vehicle might have arrived
            return np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

    def step(self, action):
        reward = 0.0
        terminated = False
        truncated = False
        
        try:
            dist_to_dest = traci.vehicle.getDrivingDistance(self.ego_vehicle_id, self.destination_edge, 0.0)
            curr_edge = traci.vehicle.getRoadID(self.ego_vehicle_id)
            
            # Apply strict 300m rule and thrashing penalty from the architecture plan
            is_switch = (action != 0)
            
            if is_switch:
                self.total_reroutes += 1
                if dist_to_dest < 300.0 and dist_to_dest > 0: # Do not reroute under 300m
                    reward -= 10.0 # Proximity Penalty
                else:
                    # Phase 2: Criticality Adjusts Thrashing Penalty
                    # High criticality responders ignore thrashing costs to save every second
                    penalty = 0.2 if self.criticality_level == "High" else 1.0
                    reward -= penalty 
            
            # Attempt to apply routing action
            if is_switch:
                # In SUMO, traci.vehicle.rerouteTraveltime computes a new optimal route based on latest edge travel times
                # For an explicit DQN discrete action, we simulate choosing 'Alternative A' by triggering a reroute.
                # A more complex setup would provide explicit edge paths as actions.
                try:
                    traci.vehicle.rerouteTraveltime(self.ego_vehicle_id)
                except Exception:
                    pass

            # Advance simulation by 15 simulation seconds (assuming 1 step = 1 sec)
            for _ in range(15):
                if self.ego_vehicle_id not in traci.vehicle.getIDList():
                    terminated = True
                    break
                
                speed = traci.vehicle.getSpeed(self.ego_vehicle_id)
                self.total_travel_time += 1.0
                if speed < 0.1:
                    self.total_waiting_time += 1.0
                    
                # Phase 2: Apply Emergency Measures (Pre-emption & Sirens)
                self._apply_emergency_measures()
                
                traci.simulationStep()
                self.sim_step += 1
            
            # Final check in case the vehicle left on the exact last step of the loop
            if not terminated and self.ego_vehicle_id not in traci.vehicle.getIDList():
                terminated = True
                
            if not terminated:
                # Positive reward for moving closer/reducing travel time (using speed as proxy)
                speed = traci.vehicle.getSpeed(self.ego_vehicle_id)
                reward += (speed / 13.89) # Normalize by ~50km/h
                
                # Bonus reward for emergency vehicle speed maintenance
                if speed > 10.0:
                    reward += 1.0
                
            else:
                # Vehicle successfully arrived!
                reward += 100.0

        except traci.exceptions.TraCIException:
            terminated = True
            
        if self.sim_step >= self.max_steps:
            truncated = True

        info = {}
        if terminated or truncated:
            info = {
                "avg_travel_time": self.total_travel_time,
                "avg_waiting_time": self.total_waiting_time,
                "reroutes": self.total_reroutes
            }

        obs = self._get_obs()
        return obs, reward, terminated, truncated, info

    def _apply_emergency_measures(self):
        """
        Phase 2: Core emergency features simulation using TraCI.
        Involves Traffic Light Pre-emption and Siren yielding logic.
        """
        if self.ego_vehicle_id not in traci.vehicle.getIDList():
            return

        # 1. Traffic Light Pre-emption
        # Identify upcoming traffic lights
        try:
            next_tls = traci.vehicle.getNextTLS(self.ego_vehicle_id)
            for tls_id, index, distance, state in next_tls:
                if distance < 100: # Within 100m of intersection
                    # Force the traffic light to green for the current edge
                    # We look for a state that allows green (usually 'G' or 'g' in that index)
                    current_logic = traci.trafficlight.getLogic(tls_id)
                    # For simplicity in this demo, we use setPhase to a known green phase or force state
                    # A robust implementation would calculate the exact phase index for the direction
                    # Here we force a generic override state if available
                    traci.trafficlight.setPhaseDuration(tls_id, 10) 
        except:
            pass

        # 2. Siren Simulation (Yielding)
        # Find vehicles on the same edge or neighboring lanes
        try:
            curr_edge = traci.vehicle.getRoadID(self.ego_vehicle_id)
            vehs_on_edge = traci.edge.getLastStepVehicleIDs(curr_edge)
            
            for veh_id in vehs_on_edge:
                if veh_id != self.ego_vehicle_id:
                    # Simulation: Force vehicles to slow down or move to right lane to simulate 'pulling over'
                    # Lane change mode 0 means no autonomous changes
                    traci.vehicle.setLaneChangeMode(veh_id, 0)
                    traci.vehicle.setSpeedMode(veh_id, 0) # Disable automatic speed control
                    traci.vehicle.slowDown(veh_id, 2.0, 5) # Slow to 2m/s over 5 seconds
        except:
            pass

    def close(self):
        try:
            traci.close()
        except:
            pass
