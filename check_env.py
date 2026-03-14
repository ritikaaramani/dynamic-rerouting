import os
import sys

def check_env():
    if 'SUMO_HOME' not in os.environ:
        print("ERROR: SUMO_HOME environment variable is not set.")
        print("Please install SUMO from https://eclipse.dev/sumo/ and set SUMO_HOME.")
        sys.exit(1)
    
    sumo_home = os.environ['SUMO_HOME']
    tools = os.path.join(sumo_home, 'tools')
    if tools not in sys.path:
        sys.path.append(tools)
    
    try:
        import traci
        print(f"SUCCESS: traci imported successfully from {tools}")
    except ImportError:
        print("ERROR: Could not import traci from SUMO_HOME/tools.")
        sys.exit(1)

if __name__ == "__main__":
    check_env()
