import os
import subprocess
import sys

def get_sumo_tools():
    if 'SUMO_HOME' not in os.environ:
        # Fallback to checking the system level SUMO_HOME if it wasn't exported in this Python env
        sys.exit("ERROR: SUMO_HOME environment variable is not set. Cannot run SUMO tools.")
    return os.path.join(os.environ['SUMO_HOME'], 'tools')

def generate_network(output_dir):
    os.makedirs(output_dir, exist_ok=True)
    net_file = os.path.join(output_dir, "grid.net.xml")
    
    # Generate a 3x3 grid network with 200m edge length
    print(f"Generating network at {net_file}...")
    subprocess.run([
        "netgenerate", 
        "--grid", 
        "--grid.number=3", 
        "--grid.length=200", 
        "--output-file", net_file
    ], check=True)
    return net_file

def generate_routes(net_file, output_dir):
    tools_dir = get_sumo_tools()
    random_trips_script = os.path.join(tools_dir, "randomTrips.py")
    routes_file = os.path.join(output_dir, "grid.rou.xml")
    
    print(f"Generating routes at {routes_file}...")
    # Generate random trips for 1000 seconds, emitting a vehicle every 2 seconds
    subprocess.run([
        sys.executable, random_trips_script,
        "-n", net_file,
        "-r", routes_file,
        "-e", "1000",
        "-p", "2"
    ], check=True)
    return routes_file

def generate_sumocfg(net_file, routes_file, output_dir):
    cfg_file = os.path.join(output_dir, "grid.sumocfg")
    print(f"Generating SUMO config at {cfg_file}...")
    cfg_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<configuration xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:noNamespaceSchemaLocation="http://sumo.dlr.de/xsd/sumoConfiguration.xsd">
    <input>
        <net-file value="{os.path.basename(net_file)}"/>
        <route-files value="{os.path.basename(routes_file)}"/>
    </input>
    <time>
        <begin value="0"/>
        <end value="1000"/>
    </time>
</configuration>
"""
    with open(cfg_file, "w") as f:
        f.write(cfg_content)
    return cfg_file

if __name__ == "__main__":
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    try:
        net = generate_network(data_dir)
        rou = generate_routes(net, data_dir)
        cfg = generate_sumocfg(net, rou, data_dir)
        print("Successfully generated all SUMO configuration files!")
    except Exception as e:
        print(f"Error generating files: {e}")
        sys.exit(1)
