# experiments/run_baselines.py

import os
import sys
import numpy as np
import tensorflow as tf
from src.env.ioht_scheduler_env import IoHTSchedulerEnv
from src.utils.config_loader import ConfigLoader
from src.utils.data_preprocessor import DataPreprocessor
# Import simplified DRL network/agent if available (e.g., from an external library)
# from baselines.gcn_drl_agent import GCN_DRL_Agent # Example baseline DRL agent

# --- Define Simple Baseline Schedulers ---

def first_come_first_served(env, state):
    """
    Implements a simple First-Come, First-Served (FCFS) scheduling policy.
    Ignores task priority and dependencies, simply schedules the oldest task 
    onto the first available resource. (Non-ML Baseline)
    """
    # Logic to select the oldest task in the queue and an available resource.
    # Returns the action index required by env.step()
    pass


def critical_first_greedy(env, state):
    """
    Implements a greedy heuristic that always prioritizes tasks based on 
    criticality or earliest deadline. (Heuristic Baseline)
    """
    # Logic to determine the highest priority task and assign it optimally 
    # based on current resource loads.
    # Returns the action index.
    pass


# --- Define DRL Baseline Class ---

class GNN_DRL_Baseline:
    """
    Represents a simpler, non-Transformer DRL baseline, like a standard 
    Graph Convolutional Network (GCN) or a simple MLP DRL (CGNN-DRL).
    """
    def __init__(self, config, env_params):
        # Initialize a simpler policy network (e.g., standard GNN or MLP)
        self.policy_net = self._build_simplified_network(config, env_params)
        # Load pre-trained weights for this baseline if available
        # self.policy_net.load_weights(...)

    def _build_simplified_network(self, config, env_params):
        # Defines the architecture for the non-SGT baseline (e.g., GCN + Dense Layers)
        pass

    def get_action(self, state_tensors):
        # Forward pass to get action logits and select the best action
        # Returns the action index
        pass


def run_single_method(method_name, agent_or_policy, env, preprocessor, num_episodes):
    """Runs a single scheduling method for N episodes and collects metrics."""
    
    episode_metrics = []
    
    for episode in range(num_episodes):
        raw_state, _ = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            # 1. Get Action based on Method Type
            if method_name in ["FCFS", "CRITICAL_GREEDY"]:
                # Non-DRL policies take raw state data
                action = agent_or_policy(env, raw_state) 
            else:
                # DRL policies need preprocessed tensors
                state_tensors = preprocessor.preprocess_state(raw_state)
                action = agent_or_policy.get_action(state_tensors)
            
            # 2. Step Environment
            raw_state, reward, done, info = env.step(action)
            episode_reward += reward

            if done:
                episode_metrics.append({
                    'reward': episode_reward,
                    'makespan': info.get('makespan'),
                    'missed_deadlines': info.get('missed_deadlines'),
                    'method': method_name
                })
        
        print(f"  -> {method_name} Ep {episode}: Reward={episode_reward:.2f}")

    # Calculate average metrics across all episodes
    avg_metrics = {
        'avg_reward': np.mean([m['reward'] for m in episode_metrics]),
        'avg_makespan': np.mean([m['makespan'] for m in episode_metrics]),
        'total_missed_deadlines': np.sum([m['missed_deadlines'] for m in episode_metrics]),
    }
    return avg_metrics


def main():
    """Orchestrates the running of all baseline methods."""
    
    CONFIG_PATH = 'config.yaml'
    config_loader = ConfigLoader(CONFIG_PATH)
    config = config_loader.get_config()

    env = IoHTSchedulerEnv(config['env'])
    preprocessor = DataPreprocessor(CONFIG_PATH)
    num_episodes = config['experiments']['num_baseline_episodes']
    
    all_results = {}
    
    print(f"--- Starting Baseline Comparison Study (N={num_episodes} Episodes) ---")

    # --- 1. Run Non-ML/Heuristic Baselines ---
    
    # Run FCFS
    all_results["FCFS"] = run_single_method("FCFS", first_come_first_served, env, preprocessor, num_episodes)

    # Run Critical Greedy Heuristic
    all_results["CRITICAL_GREEDY"] = run_single_method("CRITICAL_GREEDY", critical_first_greedy, env, preprocessor, num_episodes)


    # --- 2. Run DRL Baselines ---
    
    # Run a simpler DRL agent (e.g., CGNN-DRL)
    # This requires separate training/loading of the baseline DRL model
    gcn_drl_agent = GNN_DRL_Baseline(config, config['env'])
    all_results["CGNN_DRL"] = run_single_method("CGNN_DRL", gcn_drl_agent, env, preprocessor, num_episodes)


    # 3. Final Comparison and Output
    print("\n\n--- BASELINE RESULTS SUMMARY ---")
    print(f"{'Method':<20} | {'Avg Reward':<10} | {'Avg Makespan':<12} | {'Total Missed Deadlines':<25}")
    print("-" * 75)
    for name, res in all_results.items():
        print(f"{name:<20} | {res['avg_reward']:<10.4f} | {res['avg_makespan']:<12.2f} | {res['total_missed_deadlines']:<25}")

    # Note: The result of the final DDRL++ model (Teacher or Student) 
    # would be added here manually from its own logs for the final comparison.
    # save_results_to_file(all_results)


if __name__ == '__main__':
    main()