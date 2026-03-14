import os
import requests
import subprocess
import sys

def get_sumo_tools():
    if 'SUMO_HOME' not in os.environ:
        sys.exit("ERROR: SUMO_HOME environment variable is not set.")
    return os.path.join(os.environ['SUMO_HOME'], 'tools')

def download_osm(output_path):
    print("Downloading Bangalore OSM data via Overpass API (Central BBox)...")
    # Bounding box for central Bangalore (Cubbon Park / MG Road area)
    # south, west, north, east
    bbox = "12.9600,77.5800,12.9800,77.6100"
    query = f"""
    [out:xml][timeout:25];
    (
      way["highway"]({bbox});
      node(w);
    );
    out body;
    """
    url = "http://overpass-api.de/api/interpreter"
    response = requests.post(url, data={'data': query})
    if response.status_code == 200:
        with open(output_path, "wb") as f:
            f.write(response.content)
        print("Download successful.")
    else:
        sys.exit(f"Failed to download OSM data: {response.status_code}")

def convert_osm_to_net(osm_file, net_file):
    print("Converting OSM to SUMO network...")
    subprocess.run([
        "netconvert",
        "--osm-files", osm_file,
        "--output-file", net_file,
        "--geometry.remove",
        "--ramps.guess",
        "--junctions.join",
        "--tls.guess-signals",
        "--tls.discard-simple",
        "--tls.join"
    ], check=True)

def generate_routes(net_file, routes_file):
    tools_dir = get_sumo_tools()
    random_trips_script = os.path.join(tools_dir, "randomTrips.py")
    
    print("Generating routes...")
    subprocess.run([
        sys.executable, random_trips_script,
        "-n", net_file,
        "-r", routes_file,
        "-e", "1000",
        "-p", "1.5"
    ], check=True)

def generate_sumocfg(net_file, routes_file, output_dir):
    cfg_file = os.path.join(output_dir, "bangalore.sumocfg")
    print("Generating SUMO config...")
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
    osm_file = os.path.join(data_dir, "bangalore.osm")
    net_file = os.path.join(data_dir, "bangalore.net.xml")
    routes_file = os.path.join(data_dir, "bangalore.rou.xml")
    
    download_osm(osm_file)
    convert_osm_to_net(osm_file, net_file)
    generate_routes(net_file, routes_file)
    generate_sumocfg(net_file, routes_file, data_dir)
    print("Successfully generated Bangalore SUMO simulation environment!")
