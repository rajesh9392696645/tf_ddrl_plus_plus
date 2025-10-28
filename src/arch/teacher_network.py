# src/arch/teacher_network.py

import tensorflow as tf
from keras.src.models import Model

# --- PLACEHOLDER IMPORTS (Ensure these classes exist in your 'src/arch' directory) ---
# NOTE: You must have SGTEncoder defined somewhere, likely in 'src/arch/sgt_encoder.py'
try:
    from .sgt_encoder import SGTEncoder 
except ImportError:
    # Use a dummy class if SGTEncoder isn't defined yet, this will crash later if not implemented
    class SGTEncoder(tf.keras.layers.Layer):
        def __init__(self, d_model, num_layers, num_heads, dff, **kwargs):
            super().__init__(**kwargs)
            # Dummy layers for initialization
            self.d_model = d_model
            self.linear = tf.keras.layers.Dense(d_model)
        
        # Placeholder call signature for graph input
        def call(self, inputs, training=False):
            node_features, _ = inputs 
            return self.linear(node_features)
# ----------------------------------------------------------------------------------

class TeacherNetwork(Model):
    """
    The large Teacher Network using a Graph Encoder (like SGT) for state representation
    and separate heads for Policy and Value output.
    """
    # FIX: Explicitly accept all custom config arguments
    def __init__(self, action_space_size, state_feature_dim, encoder_type, d_model, num_layers, num_heads, dff, **kwargs):
        # Pass only recognized Keras keyword arguments to the parent class
        super(TeacherNetwork, self).__init__(**kwargs) 

        self.action_space_size = action_space_size
        self.state_feature_dim = state_feature_dim
        
        # 1. Select and Initialize Encoder
        if encoder_type == 'SGT':
            # Initialize SGTEncoder with parameters from config
            self.encoder = SGTEncoder(d_model=d_model, num_layers=num_layers, num_heads=num_heads, dff=dff)
        else:
            raise ValueError(f"Unknown encoder type: {encoder_type}")

        # 2. Shared Decoder/Head components
        # Policy Head (Outputs logits for action selection)
        self.policy_head = tf.keras.layers.Dense(action_space_size, name="policy_output")
        # Value Head (Outputs the estimated state value)
        self.value_head = tf.keras.layers.Dense(1, name="value_output")

    def call(self, inputs, training=False):
        """Forward pass through the network."""
        # inputs is expected to be a list/tuple: [node_features, adjacency_matrix]
        node_features, adjacency_matrix = inputs
        
        # 1. Process and Encode State
        # FIX: Pass 'training=training' as a keyword argument to the encoder
        encoded_state = self.encoder([node_features, adjacency_matrix], training=training)

        # 2. Policy Output (Global Pooling + Dense Head)
        # Use mean pooling to get a single vector representation for the graph
        pooled_features = tf.reduce_mean(encoded_state, axis=1)
        
        policy_logits = self.policy_head(pooled_features)
        
        # 3. Value Output (State Value Estimate)
        state_value = self.value_head(pooled_features)
        
        return policy_logits, state_value