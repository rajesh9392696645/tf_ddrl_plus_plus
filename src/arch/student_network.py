# src/arch/student_network.py

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Layer, Flatten
# A simple GNN or MLP might be used instead of the full SGT

class SimpleFeatureProcessor(Layer):
    """
    A lightweight replacement for the SparseGraphTransformerEncoder.
    This could be a shallow Graph Neural Network (GNN) or a simple 
    Multi-Layer Perceptron (MLP) applied to flattened/aggregated features.
    The goal is speed over expressiveness.
    """
    def __init__(self, output_dim, **kwargs):
        super(SimpleFeatureProcessor, self).__init__(**kwargs)
        # Define a few simple dense layers or a very shallow GNN layer
        self.dense1 = Dense(units=output_dim // 2, activation='relu')
        self.dense2 = Dense(units=output_dim, activation='relu')

    def call(self, node_features, adjacency_matrix=None, training=False):
        # 1. Simple aggregation (e.g., global pooling) or direct use of a simple GNN.
        # This part must be much faster than the SGT.
        # Example: Simple averaging of node features if the graph structure is ignored for speed.
        if node_features.shape.rank == 3:
             # Assume input is [Batch, Num_Nodes, Feature_Size]
             aggregated_features = tf.reduce_mean(node_features, axis=1) # Global average pooling
        else:
             aggregated_features = node_features # Already flattened

        x = self.dense1(aggregated_features)
        return self.dense2(x)


class StudentNetwork(Model):
    """
    The lightweight DRL agent, trained via Knowledge Distillation (KD).
    It contains the feature extractor, and the policy/value heads.
    """
    def __init__(self, action_space_size, state_feature_dim, **kwargs):
        super(StudentNetwork, self).__init__(**kwargs)
        self.action_space_size = action_space_size
        
        # 1. Lightweight Feature Encoder
        # The key difference from the Teacher is in this component.
        self.feature_extractor = SimpleFeatureProcessor(output_dim=state_feature_dim * 2)

        # 2. Policy Head (Output: Action probabilities/logits)
        # Often a single, small Dense layer
        self.policy_head = Dense(
            units=action_space_size, 
            name='student_policy_head'
        )

        # 3. Value Head (Output: State Value estimate)
        self.value_head = Dense(
            units=1, 
            name='student_value_head'
        )

    def call(self, inputs, training=False):
        # The input is expected to be a tuple or dict of (node_features, adjacency_matrix)
        # For simplicity, we assume a single input tensor for features (node_features)
        
        # Extract features
        features = self.feature_extractor(inputs[0], inputs[1], training)
        
        # Get action logits and value estimate
        action_logits = self.policy_head(features)
        value_estimate = self.value_head(features)

        # Return the required DRL outputs
        return action_logits, value_estimate

    # Optional: A method to get the output from the policy head (for KD)
    def policy_logits(self, inputs, training=False):
        features = self.feature_extractor(inputs[0], inputs[1], training)
        return self.policy_head(features)