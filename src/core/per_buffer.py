# src/core/per_buffer.py

import numpy as np
import tensorflow as tf
from typing import NamedTuple, List

# Define the structure for a single transition/experience
Transition = NamedTuple('Transition', [
    ('state', np.ndarray), 
    ('action', np.ndarray), 
    ('reward', float), 
    ('next_state', np.ndarray), 
    ('done', bool)
])

# --- SumTree Class (Internal Helper) ---
class SumTree:
    """
    A binary tree structure used to efficiently:
    1. Store priorities (leaves).
    2. Store the sum of priorities (internal nodes).
    3. Sample an experience proportional to its priority in O(log N).
    """
    def __init__(self, capacity):
        # Capacity must be a power of 2 for a full binary tree implementation
        self.capacity = capacity
        # Tree nodes store the sum of priorities (size = 2*capacity - 1)
        self.tree = np.zeros(2 * capacity - 1) 
        # Data leaves store the actual transitions (size = capacity)
        self.data = np.zeros(capacity, dtype=object) 
        self.data_pointer = 0 # Pointer to the next empty data slot

    def _propagate(self, idx, change):
        # Update parent nodes when a leaf changes
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        # Find the index of the leaf corresponding to the value 's'
        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total_priority(self):
        # The root node holds the sum of all priorities
        return self.tree[0]

    def add(self, priority, data):
        # Store priority and data
        tree_idx = self.data_pointer + self.capacity - 1
        self.data[self.data_pointer] = data
        self.update_priority(tree_idx, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity

    def get_leaf(self, v):
        # Retrieve the leaf (index, priority, data) based on a sampled value 'v'
        tree_idx = self._retrieve(0, v)
        data_idx = tree_idx - self.capacity + 1
        return tree_idx, self.tree[tree_idx], self.data[data_idx]
    
    def update_priority(self, tree_idx, priority):
        # Update the priority value at a leaf and propagate the change
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)


# --- PrioritizedExperienceReplayBuffer Class (Main) ---
class PrioritizedExperienceReplayBuffer:
    """
    Main implementation of the PER buffer using a SumTree.
    """
    def __init__(self, capacity, alpha=0.6, beta=0.4, epsilon=1e-6):
        self.capacity = capacity
        self.alpha = alpha  # Controls the prioritization exponent (0=uniform, 1=full priority)
        self.beta = beta    # Controls the importance-sampling weight (annealed from 0.4 to 1.0)
        self.epsilon = epsilon # Small constant to ensure non-zero priority
        self.tree = SumTree(capacity)
        # Max priority is often initialized to 1.0
        self.max_priority = 1.0

    def store(self, experience: Transition):
        """
        Stores a new experience with the current maximum priority.
        """
        # New samples are stored with max priority to ensure they are sampled at least once.
        self.tree.add(self.max_priority, experience)

    def sample(self, batch_size):
        """
        Samples a batch of experiences with probability P_i = p_i^alpha / sum(p_k^alpha)
        and calculates Importance Sampling (IS) weights.
        """
        # 1. Divide the total priority range into 'batch_size' segments
        segment_len = self.tree.total_priority() / batch_size
        
        indices = []
        transitions = []
        priorities = []
        
        # 2. Sample a random value from each segment
        for i in range(batch_size):
            a = segment_len * i
            b = segment_len * (i + 1)
            v = np.random.uniform(a, b) # Sample a value within the segment
            
            # 3. Retrieve the corresponding leaf from the SumTree
            tree_idx, priority, data = self.tree.get_leaf(v)
            
            indices.append(tree_idx)
            priorities.append(priority)
            transitions.append(data)

        # Convert to numpy arrays for batch processing
        priorities = np.array(priorities)
        
        # 4. Calculate IS weights (w_i)
        # Weight formula: w_i = (N * P_i)^(-beta) / max(w_j)
        
        # Probability P_i = p_i / sum(p_k)
        prob = priorities / self.tree.total_priority()
        N = self.capacity # Use capacity as a proxy for the total number of samples
        
        # IS weights
        weights = (N * prob) ** (-self.beta)
        
        # Normalize weights by the maximum weight found in the batch (for stability)
        max_weight = np.max(weights)
        norm_weights = weights / max_weight

        # 5. Collate the batched data (states, actions, etc.)
        batch_states = np.array([t.state for t in transitions])
        batch_actions = np.array([t.action for t in transitions])
        # ... (other batched components: rewards, next_states, dones) ...
        
        return batch_states, batch_actions, ..., indices, norm_weights


    def update_priorities(self, tree_indices, td_errors):
        """
        Updates the priority of sampled experiences based on their new TD errors.
        This is the most critical step after a learning update.
        """
        new_priorities = np.abs(td_errors) + self.epsilon
        
        # Update the max priority tracker
        self.max_priority = max(self.max_priority, np.max(new_priorities))
        
        # Apply the alpha exponent and update the SumTree
        for tree_idx, priority in zip(tree_indices, new_priorities):
            # p_i = |TD_error_i|^alpha
            self.tree.update_priority(tree_idx, priority ** self.alpha)

    # Optional: Method to smoothly anneal the beta parameter
    def anneal_beta(self, current_step, total_steps):
        # Beta often starts at 0.4 and is annealed to 1.0 over training
        self.beta = min(1.0, self.beta + (1.0 - 0.4) / total_steps)

    def __len__(self):
        # Provides the number of stored experiences
        return self.tree.data_pointer