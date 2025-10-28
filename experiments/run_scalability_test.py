# experiments/run_scalability_test.py

import os
import sys
import copy
import numpy as np
import time
from src.utils.config_loader import ConfigLoader
from src.env.ioht_scheduler_env import IoHTSchedulerEnv
from src.utils.data_preprocessor import DataPreprocessor
from src.arch.student_network import StudentNetwork
from main_edge_deployer import run_single_inference_episode # Assuming this helper exists

# --- Define Scalability Parameters ---

# Define the variable to scale and the test points
# Scaling Factor: Number of Tasks (N) or Number of Resources (M)
SCALING_FACTOR_KEY = 'max_tasks'

# List of values for the scaling factor (e.g., small, medium, large loads)
SCALING_TEST_POINTS = [
    10,  # Light Load
    20,  # Medium Load
    30,  # Heavy Load
    40   # Extreme Load
]

def load_student_agent(config, env_params):
    """Initializes and loads the distilled Student Network weights."""
    # (Same loading logic as in main_edge_deployer.py)
    student_agent = StudentNetwork(...)
    # student_agent.load_weights(...).expect_partial()
    return student_agent

def evaluate_at_scale(scale_value, student_agent, base_config, num_runs=50):
    """
    Runs the student agent N times at a specific scale value and collects metrics.
    
    :param scale_value: The current number of tasks/resources to test.
    """
    
    # 1. Temporarily modify the configuration for the environment
    scaled_config = copy.deepcopy(base_config)
    scaled_config['env'][SCALING_FACTOR_KEY] = scale_value
    
    # 2. Re-initialize Environment and Preprocessor with the new size
    env = IoHTSchedulerEnv(scaled_config['env'])
    preprocessor = DataPreprocessor('config.yaml') # Must be consistent
    
    # 3. Run Evaluation Loops
    all_rewards = []
    all_inference_times = []
    all_makespans = []

    for run in range(num_runs):
        raw_state, _ = env.reset()
        done = False
        episode_rewards = []
        inference_times_episode = []
        
        while not done:
            # --- Inference and Timing ---
            state_tensors = preprocessor.preprocess_state(raw_state)
            start_time = time.time()
            
            # Agent Decision (Forward Pass)
            logits, _ = student_agent(state_tensors, training=False)
            action = tf.argmax(logits, axis=-1).numpy()[0]
            
            inference_times_episode.append((time.time() - start_time) * 1000) # ms

            # --- Environment Step ---
            raw_state, reward, done, info = env.step(action)
            episode_rewards.append(reward)

            if done:
                all_rewards.append(np.sum(episode_rewards))
                all_inference_times.append(np.mean(inference_times_episode))
                all_makespans.append(info.get('makespan', 0))

    # 4. Aggregate Results
    results = {
        'scale_value': scale_value,
        'avg_reward': np.mean(all_rewards),
        'avg_makespan': np.mean(all_makespans),
        'avg_inference_ms': np.mean(all_inference_times)
    }
    return results


def main():
    """Orchestrates the running of scalability tests across all defined test points."""
    
    CONFIG_PATH = 'config.yaml'
    base_config = ConfigLoader(CONFIG_PATH).get_config()
    
    # Load the trained Student Agent only once
    student_agent = load_student_agent(base_config, base_config['env'])

    all_scalability_results = []
    
    print(f"--- Starting Scalability Test on {SCALING_FACTOR_KEY} ---")

    for scale_point in SCALING_TEST_POINTS:
        print(f"\nEvaluating Scale Point: {SCALING_FACTOR_KEY}={scale_point}")
        
        # Run the evaluation for the current scale point
        results = evaluate_at_scale(scale_point, student_agent, base_config, num_runs=50)
        all_scalability_results.append(results)
        
        print(f"Results: Avg Reward={results['avg_reward']:.4f}, Avg Inference Time={results['avg_inference_ms']:.3f}ms")


    # 4. Final Summary and Output
    print("\n\n--- SCALABILITY TEST SUMMARY ---")
    print(f"{SCALING_FACTOR_KEY:<15} | {'Avg Reward':<12} | {'Avg Makespan':<12} | {'Avg Inference Time (ms)':<25}")
    print("-" * 70)
    
    for res in all_scalability_results:
        print(f"{res['scale_value']:<15} | {res['avg_reward']:<12.4f} | {res['avg_makespan']:<12.2f} | {res['avg_inference_ms']:<25.3f}")

    # Logic to save results to a CSV/JSON file for plotting (Chapter 8 figures)
    # save_results_to_file(all_scalability_results)


if __name__ == '__main__':
    main()