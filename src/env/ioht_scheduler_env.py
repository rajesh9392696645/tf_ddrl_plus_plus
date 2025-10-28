# src/env/ioht_scheduler_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import random
import time
from typing import Dict, Any, Tuple

# --- Constants (Define these based on your config.yaml) ---
MAX_TASKS = 30
MAX_RESOURCES = 10
# (Other constants like feature dimensions, etc.)


class IoHTSchedulerEnv(gym.Env):
    """
    Custom Environment for scheduling IoHT tasks onto edge/cloud resources.
    The state is represented as a dynamic graph.
    """
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, env_config: Dict[str, Any]):
        super().__init__()
        
        # Load parameters from the config dictionary
        self.max_tasks = env_config.get('max_tasks', MAX_TASKS)
        self.max_resources = env_config.get('max_resources', MAX_RESOURCES)
        self.action_space_size = env_config.get('action_space_size', self.max_tasks * self.max_resources)
        
        # --- Define Spaces ---
        self.action_space = spaces.Discrete(self.action_space_size)
        
        # Observation Space (Conceptual definition for a Graph State)
        # Note: The actual check for the Observation Space is often skipped 
        # for complex graph inputs, but a placeholder is necessary.
        self.observation_space = spaces.Dict({
            'node_features': spaces.Box(low=-np.inf, high=np.inf, shape=(self.max_tasks + self.max_resources, 15), dtype=np.float32),
            'adj_matrix': spaces.Box(low=0, high=1, shape=(self.max_tasks + self.max_resources, self.max_tasks + self.max_resources), dtype=np.float32),
            # Add other state components needed by your DRL model
        })

        # --- Internal State ---
        self.task_queue = []
        self.resource_status = {}
        self.current_time = 0.0
        # self.current_graph_state will hold the internal NetworkX or dictionary representation
        self.current_graph_state = None 

    # ------------------------------------------------------------------
    # CORE ENVIRONMENT METHODS
    # ------------------------------------------------------------------

    def reset(self, seed: int = None, options: Dict[str, Any] = None) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """
        Resets the environment. 
        CRITICAL FIX: Ensures the observation is a dictionary structure.
        """
        super().reset(seed=seed)
        
        # 1. Initialize Internal State
        # Reset task queue, resource statuses, and environment time
        self.task_queue = self._generate_initial_tasks()
        self.resource_status = self._initialize_resources()
        self.current_time = 0.0
        
        # 2. Build the Initial Graph State (Internal representation)
        self.current_graph_state = self._build_internal_graph()
        
        # 3. Generate the Observation Dictionary (The expected output format)
        observation = self._get_observation() 
        
        # 4. Generate Info Dictionary (Optional metadata)
        info = {
            'tasks_in_queue': len(self.task_queue),
            'current_time': self.current_time
        }
        
        # This return structure is what main_cloud_trainer.py now expects:
        # (observation_dictionary, info_dictionary)
        return observation, info

    def step(self, action: int) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """
        Applies a scheduling action and advances the simulation.
        """
        # ... (Step logic: decode action, schedule task, update time)
        
        # 1. Update internal state
        # ...
        
        # 2. Compute Reward and Termination
        reward = 0.0
        terminated = False # True if all tasks are finished or max_time reached
        truncated = False # True if episode length exceeds limit

        # 3. Generate Next Observation
        observation = self._get_observation()

        # 4. Generate Info
        info = {}

        # Gym/Gymnasium requires 5 return values now: obs, reward, terminated, truncated, info
        return observation, reward, terminated, truncated, info

    # ------------------------------------------------------------------
    # HELPER METHODS (Conceptual)
    # ------------------------------------------------------------------

    def _get_observation(self) -> Dict[str, np.ndarray]:
        """
        Converts the internal state (self.current_graph_state) into 
        the structured numpy array dictionary required by the DRL agent.
        """
        
        # *** CRITICAL: The keys here must match what is accessed in main_cloud_trainer.py ***
        return {
            # This MUST be a numpy array of shape (N+M, F) 
            'node_features': self._compute_node_features(), 
            
            # This is typically an Adjacency Matrix (N+M, N+M)
            'adj_matrix': self._compute_adjacency_matrix(),
        }

    def _generate_initial_tasks(self):
        """Generates a list of initial tasks with required attributes."""
        # ... (implementation) ...
        return []

    def _initialize_resources(self):
        """Sets up the initial status of all resources."""
        # ... (implementation) ...
        return {}
    
    def _build_internal_graph(self):
        """Creates the internal graph representation (e.g., NetworkX object)."""
        # ... (implementation) ...
        return {} # Or a NetworkX object

    def _compute_node_features(self) -> np.ndarray:
        """Calculates the feature vector for every node (task and resource)."""
        # Example dummy array for features (N+M rows, F features)
        # The number of features (F) must match what your DRL model expects.
        num_nodes = self.max_tasks + self.max_resources
        # Placeholder size: 15 features 
        return np.zeros((num_nodes, 15), dtype=np.float32)

    def _compute_adjacency_matrix(self) -> np.ndarray:
        """Calculates the adjacency matrix (A) from the current graph state."""
        # Example dummy adjacency matrix
        num_nodes = self.max_tasks + self.max_resources
        return np.eye(num_nodes, dtype=np.float32)


# End of File