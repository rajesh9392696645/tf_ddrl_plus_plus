# main_edge_deployer.py

import os
import sys
import time
import numpy as np
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src'))) 

# Import core components
from src.arch.student_network import StudentNetwork
from src.env.ioht_scheduler_env import IoHTSchedulerEnv
from src.utils.config_loader import ConfigLoader
from src.utils.data_preprocessor import DataPreprocessor

def main():
    """
    Main function to load the distilled Student Agent and run it in the 
    IoHT environment for real-time inference and evaluation. (Chapter 6 Logic)
    """
    
    # 1. Configuration and Setup
    try:
        CONFIG_PATH = 'config.yaml'
        config_loader = ConfigLoader(CONFIG_PATH)
        config = config_loader.get_config()
    except Exception as e:
        print(f"FATAL: Failed to load configuration. Error: {e}")
        sys.exit(1)

    print("Starting Edge Deployment Simulation for Student Agent.")

    # 2. Environment and Data Preprocessor Initialization
    env_params = config['env']
    env = IoHTSchedulerEnv(env_params)
    preprocessor = DataPreprocessor(CONFIG_PATH)

    # 3. Student Model Initialization and Weight Loading
    # Define model dimensions
    action_space_size = env_params['action_space_size']
    state_feature_dim = config['arch']['d_model'] # Use the expected feature dimension
    
    student_agent = StudentNetwork(
        action_space_size=action_space_size, 
        state_feature_dim=state_feature_dim,
        **config['arch']
    )

    # Load the trained/distilled weights for the Student
    student_weights_path = config['distillation']['final_student_model_path']
    try:
        # Build the model by calling it once with dummy data before loading weights
        # (A standard requirement for Keras/TF subclassed models)
        dummy_state, _ = env.reset()
        dummy_inputs = preprocessor.preprocess_state(dummy_state)
        student_agent(dummy_inputs) 
        
        student_agent.load_weights(student_weights_path).expect_partial()
        print(f"Successfully loaded Student model from: {student_weights_path}")
    except Exception as e:
        print(f"ERROR: Failed to load Student weights. Ensure path is correct. Error: {e}")
        sys.exit(1)


    # 4. Inference and Evaluation Loop
    num_episodes = config['edge_deploy']['num_evaluation_episodes']
    total_metrics = [] # To track latency, makespan, and inference time

    for episode in range(num_episodes):
        raw_state, _ = env.reset()
        done = False
        episode_reward = 0
        inference_times = []
        
        while not done:
            # --- Inference (Real-Time Scheduling Decision) ---
            
            # 1. Preprocess State (Fast transformation to tensor)
            state_tensors = preprocessor.preprocess_state(raw_state)
            
            # 2. Measure Inference Time
            start_time = time.time()
            
            # 3. Agent Decision (Forward Pass of the lightweight Student Network)
            # The Student is fast, and we only need the policy logits for the decision.
            logits, _ = student_agent(state_tensors, training=False)
            
            # 4. Select Action (Greedy or sampled policy decision)
            action = tf.argmax(logits, axis=-1).numpy()[0]
            
            # 5. Record Time
            end_time = time.time()
            inference_times.append((end_time - start_time) * 1000) # ms

            # --- Environment Step ---
            raw_state, reward, done, info = env.step(action)
            episode_reward += reward
            
            if done:
                # Add final metrics like makespan, deadline misses (from info)
                avg_inf_time = np.mean(inference_times) if inference_times else 0
                total_metrics.append({
                    'episode': episode,
                    'reward': episode_reward,
                    'avg_inference_ms': avg_inf_time,
                    'makespan': info.get('makespan'),
                    'missed_deadlines': info.get('missed_deadlines')
                })
                print(f"Ep {episode}: Reward={episode_reward:.2f}, Inf Time={avg_inf_time:.3f}ms")

    # 5. Final Reporting
    avg_inf_across_all_runs = np.mean([m['avg_inference_ms'] for m in total_metrics])
    avg_reward_across_all_runs = np.mean([m['reward'] for m in total_metrics])
    
    print("\n--- Edge Deployment Results ---")
    print(f"Total Episodes: {num_episodes}")
    print(f"Average Reward: {avg_reward_across_all_runs:.2f}")
    print(f"**Avg Inference Time:** {avg_inf_across_all_runs:.3f} ms (Target Metric)")

if __name__ == '__main__':
    main()