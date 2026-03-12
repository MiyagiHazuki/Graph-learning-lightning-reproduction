import os
import sys
import argparse

# Add the project root to the path so we can import from src
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

from src.config_loader import Config
from pipeline import Pipeline

def main():
    parser = argparse.ArgumentParser(description='ProGNN')
    parser.add_argument('--config', type=str, default=os.path.join('config', 'citeseer.yaml'),
                        help='Path to the configuration file')
    args_cli = parser.parse_args()
    
    args = Config(args_cli.config)
    pipeline = Pipeline(args)
    pipeline.run()
    
if __name__ == "__main__":
    main()
