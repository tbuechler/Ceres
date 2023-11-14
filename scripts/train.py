import os
import sys
# Add parent directory to make current script easily runnable
sys.path.append(os.path.join(os.path.dirname(__file__), r'..'))

import hydra
from source.Utility.config import validate_configuration_file, process_config_file
from source.Agent import * # Fetch all agent classes

# Root of project
project_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# Change config_path and config_name according to your needs!!!
@hydra.main(config_path="../configs/tests", config_name="Overfit_PIDNetS_Cityscape_Semseg", version_base="1.3.2")
def main(cfg):
    ## Precheck of configuration file
    validate_configuration_file(cfg, 'train')

    ## Process configuration file, i.e. create logging directories
    process_config_file(cfg, root=project_root)

    ## Create new Agent instance
    agent_class = globals()[cfg.agent.name]
    agent = agent_class(cfg, mode='train')

    ## Start training process
    try:
        agent._start_training()
    except KeyboardInterrupt:
        pass
    finally:
        agent._finalize()

if __name__ == '__main__':
    main()