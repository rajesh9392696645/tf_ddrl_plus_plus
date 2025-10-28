# src/utils/data_preprocessor.py

import numpy as np
import tensorflow as tf
from ..utils.config_loader import ConfigLoader

class DataPreprocessor:
    """
    Handles the conversion of raw task and resource data from the 
    IoHTSchedulerEnv into graph tensor format (Node Features and Adjacency Matrix) 
    for the Graph-Transformer network.
    """
    
    def __init__(self, config_path):
        self.config = ConfigLoader(config_path).get_config()
        self.num_tasks = self.config['env']['max_tasks']
        self.num_resources = self.config['env']['max_resources']
        # The total number of nodes in the graph (Tasks + Resources)
        self.num_graph_nodes = self.num_tasks + self.num_resources
        
        # Define the dimensionality of the feature vectors for tasks and resources
        self.task_feature_dim = self.config['data_preprocessor']['task_feature_dim']
        self.resource_feature_dim = self.config['data_preprocessor']['resource_feature_dim']
        # The final embedding dimension for all nodes
        self.node_embedding_dim = max(self.task_feature_dim, self.resource_feature_dim) 

    
    def preprocess_state(self, raw_state_data):
        """
        Main function to take the raw environment state and produce graph tensors.
        
        :param raw_state_data: A dictionary or object containing current task/resource statuses.
        :return: A tuple of (node_features_tensor, adjacency_matrix_tensor)
        """
        # 1. Create Node Feature Matrix
        node_features = self._create_node_features(
            raw_state_data['tasks'], 
            raw_state_data['resources']
        )
        
        # 2. Create Adjacency Matrix
        adj_matrix = self._create_adjacency_matrix(
            raw_state_data['dependencies'], 
            raw_state_data['current_assignments']
        )
        
        # 3. Convert to TensorFlow tensors (or ensure they are ready for the network)
        node_features_tensor = tf.convert_to_tensor(node_features, dtype=tf.float32)
        adj_matrix_tensor = tf.convert_to_tensor(adj_matrix, dtype=tf.float32)

        # The tensors should be batched [1, Num_Nodes, Features] and [1, Num_Nodes, Num_Nodes]
        # if the DRL agent expects a batched input.
        return tf.expand_dims(node_features_tensor, axis=0), tf.expand_dims(adj_matrix_tensor, axis=0)

    
    def _create_node_features(self, tasks_data, resources_data):
        """
        Generates the feature vectors for all nodes (tasks and resources).
        """
        # Initialize the final feature matrix
        feature_matrix = np.zeros((self.num_graph_nodes, self.node_embedding_dim), dtype=np.float32)
        
        # --- Task Node Features (Indices 0 to num_tasks - 1) ---
        for i, task in enumerate(tasks_data):
            # Example Features (Criticality, Remaining Time, Deadline, Status)
            features = [
                task['is_critical'], 
                task['remaining_time'] / self.config['env']['max_time'],
                task['deadline'] / self.config['env']['max_time'],
                task['status_one_hot'] 
            ]
            # Pad or slice to match the self.node_embedding_dim
            feature_matrix[i, :len(features)] = features
            
        # --- Resource Node Features (Indices num_tasks to end) ---
        resource_start_index = self.num_tasks
        for j, resource in enumerate(resources_data):
            # Example Features (Load, Capacity, Type)
            features = [
                resource['current_load'] / resource['max_capacity'], 
                resource['resource_type_one_hot'],
                resource['availability_status']
            ]
            feature_matrix[resource_start_index + j, :len(features)] = features

        return feature_matrix

    
    def _create_adjacency_matrix(self, dependencies, current_assignments):
        """
        Generates the adjacency matrix (A) representing connections between nodes.
        The matrix size is (Num_Nodes x Num_Nodes).
        """
        # A matrix where 1 = connection, 0 = no connection
        adj_matrix = np.zeros((self.num_graph_nodes, self.num_graph_nodes), dtype=np.float32)
        
        # --- 1. Task-Task Edges (Dependencies) ---
        # Edge exists if Task A must finish before Task B starts (precedence constraint).
        for (task_a_idx, task_b_idx) in dependencies:
            # Set edges (may be directed, e.g., A -> B)
            adj_matrix[task_a_idx, task_b_idx] = 1.0 
        
        # --- 2. Task-Resource Edges (Current Assignments) ---
        # Edge exists if Task A is currently assigned to Resource R.
        resource_start_index = self.num_tasks
        for task_idx, resource_idx in current_assignments:
            res_node_idx = resource_start_index + resource_idx
            
            # Set bi-directional edges (Task <-> Resource)
            adj_matrix[task_idx, res_node_idx] = 1.0
            adj_matrix[res_node_idx, task_idx] = 1.0
            
        # --- 3. Self-Loops ---
        # Often included in GNNs/Transformers (A = I + A)
        np.fill_diagonal(adj_matrix, 1.0)

        # Note: The SGT will use this matrix for masked attention.
        return adj_matrix