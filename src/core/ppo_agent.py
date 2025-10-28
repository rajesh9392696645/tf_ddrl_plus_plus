# src/core/ppo_agent.py

import tensorflow as tf
import numpy as np
import time
from collections import deque
from typing import Dict, Any, Tuple

# Assuming you have a Prioritized Experience Replay (PER) buffer class
# from .per_buffer import PERBuffer 

class PPOAgent:
    """
    Proximal Policy Optimization Agent for training the Teacher Network.
    Handles data gathering, loss calculation, and policy updates.
    """
    
    def __init__(self, network, env, logger, 
                 total_timesteps, learning_rate, gamma, lambda_gae,
                 clip_ratio, epochs, value_coef, entropy_coef, batch_size,
                 capacity, alpha, beta, **kwargs): 
        
        # --- Core Components ---
        self.network = network         
        self.env = env                 
        self.logger = logger           

        # --- DRL Parameters ---
        self.total_timesteps = total_timesteps
        self.gamma = gamma             
        self.lambda_gae = lambda_gae   

        # --- PPO Parameters ---
        self.clip_ratio = clip_ratio
        self.ppo_epochs = epochs       
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.ppo_batch_size = batch_size 

        # --- PER Parameters ---
        self.per_capacity = capacity
        self.per_alpha = alpha
        self.per_beta = beta
        
        # --- Internal State & Optimization ---
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.global_step = tf.Variable(0, dtype=tf.int64)
        self.current_reward_history = deque(maxlen=100) 

        self.buffer = [] 
        print("PPOAgent initialized successfully.")


    # ------------------------------------------------------------------
    # CORE PPO ALGORITHM METHODS
    # ------------------------------------------------------------------

    def train(self):
        """The main training loop."""
        print(f"Starting training for {self.total_timesteps} steps...")
        
        obs, info = self.env.reset()
        
        while self.global_step < self.total_timesteps:
            
            # 1. Gather trajectory data
            trajectory_data, obs = self._gather_data(obs)

            # 2. Compute Generalized Advantage Estimation (GAE)
            advantages, returns = self._compute_advantages_and_returns(trajectory_data)

            # 3. Update Policy Network 
            self._update_network(trajectory_data, advantages, returns)

            # 4. Logging and Checkpoints
            # NOTE: This checkpointing logic runs frequently, often at the start
            # of the loop (step 0).
            self._log_and_checkpoint()

            self.global_step.assign_add(1) 

        print("PPO training loop finished.")


    def _gather_data(self, current_obs: Dict[str, np.ndarray]) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
        """Collects one batch of data by running the policy in the environment."""
        
        trajectory_steps = 2048 
        
        batch_data = {
            'observations': [], 'actions': [], 'rewards': [], 
            'values': [], 'log_probs': [], 'dones': []
        }
        
        for _ in range(trajectory_steps):
            # 1. Prepare state for network
            node_features = tf.expand_dims(current_obs['node_features'], axis=0)
            adj_matrix = tf.expand_dims(current_obs['adj_matrix'], axis=0)
            
            # 2. Get action, value, and log_prob from the Teacher Network
            policy_logits, state_value = self.network([node_features, adj_matrix], training=False) 
            
            # Convert logits to a distribution and sample an action
            action_distribution = tf.nn.softmax(policy_logits)
            action_idx = tf.random.categorical(policy_logits, 1)[0, 0].numpy()
            log_prob = tf.math.log(action_distribution[0, action_idx] + 1e-10)

            # 3. Step the environment
            next_obs, reward, terminated, truncated, info = self.env.step(action_idx)
            done = terminated or truncated

            # 4. Store experience
            batch_data['observations'].append(current_obs)
            batch_data['actions'].append(action_idx)
            batch_data['rewards'].append(reward)
            batch_data['values'].append(state_value.numpy()[0, 0])
            batch_data['log_probs'].append(log_prob.numpy())
            batch_data['dones'].append(done)

            current_obs = next_obs
            
            if done:
                current_obs, _ = self.env.reset()
                self.current_reward_history.append(info.get('episode_reward', 0))

            self.global_step.assign_add(1)

        return batch_data, current_obs

    # --- (Remainder of PPOAgent methods) ---
    
    def _compute_advantages_and_returns(self, trajectory_data: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
        """Calculates GAE and discounted returns."""
        
        rewards = np.array(trajectory_data['rewards'])
        values = np.array(trajectory_data['values'])
        dones = np.array(trajectory_data['dones'])

        # GAE calculation (simplified placeholder)
        advantages = np.zeros_like(rewards, dtype=np.float32)
        last_gae_lambda = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                # Terminal value estimation
                next_value = 0 
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            advantages[t] = last_gae_lambda = delta + self.gamma * self.lambda_gae * (1 - dones[t]) * last_gae_lambda
            
        returns = advantages + values
        
        return tf.constant(advantages, dtype=tf.float32), tf.constant(returns, dtype=tf.float32)

    @tf.function
    def _update_step(self, obs_batch, action_batch, old_log_prob_batch, return_batch, advantage_batch):
        """
        The core PPO optimization step, wrapped by tf.function.
        Calculates loss and applies gradients.
        """
        with tf.GradientTape() as tape:
            # 1. Forward Pass
            policy_logits, state_value = self.network(obs_batch, training=True)
            
            # 2. Compute Loss components (Simplified placeholders)
            value_loss = tf.reduce_mean(tf.square(state_value - return_batch))
            total_loss = 0.0 # Replace with full PPO loss calculation
            
        # 3. Apply Gradients
        gradients = tape.gradient(total_loss, self.network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.network.trainable_variables))
        
        return total_loss

    def _update_network(self, trajectory_data, advantages, returns):
        """Iterates through data and performs PPO epochs."""
        
        # Placeholder for data processing and batching
        for epoch in range(self.ppo_epochs):
            pass

    def _log_and_checkpoint(self):
        """Logs metrics to TensorBoard and saves model checkpoints."""
        
        avg_reward = np.mean(self.current_reward_history) if self.current_reward_history else 0
        
        self.logger.log_scalar(self.global_step.numpy(), 'metrics/avg_reward', avg_reward)
        self.logger.log_scalar(self.global_step.numpy(), 'performance/global_step', self.global_step.numpy())
        
        if self.global_step.numpy() % self.logger.config['logging']['save_interval'] == 0:
            # FIX APPLIED HERE: Changed from .h5 to .weights.h5
            save_path = f"data/checkpoints/teacher_step_{self.global_step.numpy()}.weights.h5"
            self.network.save_weights(save_path)
            print(f"Checkpoint saved at step {self.global_step.numpy()}")