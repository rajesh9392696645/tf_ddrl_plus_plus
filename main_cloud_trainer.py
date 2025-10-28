# main_cloud_trainer.py

import os
import sys
import time
import numpy as np
import tensorflow as tf

# Add the project root to the path for correct relative imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# Import core components
from src.arch.teacher_network import TeacherNetwork
from src.core.ppo_agent import PPOAgent
from src.env.ioht_scheduler_env import IoHTSchedulerEnv
from src.utils.config_loader import ConfigLoader
from src.utils.logging_setup import LoggingSetup

def main():
    """
    Main function to initialize and train the Teacher Agent using PPO 
    on the IoHT Scheduler Environment. (Chapter 5 Logic)
    """
    
    # 1. Configuration and Setup
    try:
        CONFIG_PATH = 'config.yaml'
        config_loader = ConfigLoader(CONFIG_PATH)
        config = config_loader.get_config()
    except Exception as e:
        print(f"FATAL: Failed to load configuration. Error: {e}")
        sys.exit(1)

    # Initialize the Logger (This has been heavily debugged and should now work)
    logger = LoggingSetup(CONFIG_PATH)
    logger.log_hparams()

    print(f"Starting Teacher Training: {logger.experiment_name}")

    # 2. Environment Initialization
    env_params = config['env']
    env = IoHTSchedulerEnv(env_params)

    # 3. Determine Network Dimensions (Requires one environment reset)
    try:
        initial_state, _ = env.reset() 
        state_feature_dim = initial_state['node_features'].shape[-1]
    except (TypeError, KeyError) as e:
        print("\n--- FATAL ENVIRONMENT ERROR ---")
        print(f"Error inspecting initial state: {e}")
        print("Please ensure IoHTSchedulerEnv.reset() returns a dictionary with the key 'node_features'.")
        sys.exit(1)


    action_space_size = env_params['action_space_size']
    
    print(f"Action Space Size: {action_space_size}")
    print(f"State Feature Dimension (from env): {state_feature_dim}")


    # 4. Teacher Model and Agent Initialization
    teacher_network = TeacherNetwork(
        action_space_size=action_space_size, 
        state_feature_dim=state_feature_dim,
        **config['arch']
    )

    # --- FIX: Create Checkpoint Directory (Resolves FileNotFoundError) ---
    checkpoint_dir = 'data/checkpoints'
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    # --------------------------------------------------------------------

    ppo_agent = PPOAgent(
        network=teacher_network,
        env=env,
        logger=logger,
        **config['drl'],
        **config['ppo'],
        **config['per']
    )

    # 5. Training Loop
    print("\nStarting PPO training loop...")
    ppo_agent.train() 

    # 6. Final Saving (Teacher model weights)
    # --- FIX: Ensure the correct Keras extension is used (.weights.h5) ---
    final_model_path = os.path.join(logger.full_log_path, "teacher_final_model.weights.h5")
    teacher_network.save_weights(final_model_path)
    # ---------------------------------------------------------------------
    
    print(f"\n--- TRAINING COMPLETE ---")
    print(f"Final Teacher model saved to: {final_model_path}")
    print(f"Review logs at: {logger.full_log_path}")

if __name__ == '__main__':
    # Increase recursion limit for potentially deep graph computations
    sys.setrecursionlimit(2000) 
    main()