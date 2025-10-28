# src/distillation/hybrid_feedback.py

import tensorflow as tf
import numpy as np
from ..utils.config_loader import ConfigLoader

class HybridFeedbackGenerator:
    """
    Logic for generating a hybrid loss signal to train the Student Network.
    This signal combines:
    1. Knowledge Distillation (KD) Loss: Student policy alignment with Teacher policy.
    2. Policy Gradient (RL) Loss: Student learning from its own experience/advantages.
    """
    def __init__(self, config_path):
        self.config = ConfigLoader(config_path).get_config()
        
        # Hyperparameters for blending the losses (Weights for the hybrid signal)
        self.kd_weight = self.config['distillation']['kd_loss_weight']   # Alpha or Beta
        self.rl_weight = self.config['distillation']['rl_loss_weight']   # 1 - kd_weight
        self.temperature = self.config['distillation']['temperature']    # T for softened KD
        
        # PPO parameters that may need to be reused/modified for the Student's RL term
        self.clip_ratio = self.config['ppo']['clip_ratio']

    
    def calculate_kd_loss(self, teacher_logits, student_logits):
        """
        Calculates the Knowledge Distillation loss, typically the KL-Divergence,
        to align the Student's policy distribution with the Teacher's "soft" target.
        """
        # 1. Soften the logits using Temperature T
        soft_teacher_probs = tf.nn.softmax(teacher_logits / self.temperature)
        soft_student_log_probs = tf.nn.log_softmax(student_logits / self.temperature)
        
        # 2. Calculate KL-Divergence (KL(P_T || P_S))
        # The Student is trained to minimize this distance.
        kl_div_loss = tf.keras.losses.KLDivergence(reduction=tf.keras.losses.Reduction.NONE)(
            soft_teacher_probs, 
            soft_student_log_probs
        )
        
        # 3. Scale the loss by T^2 (a common practice to account for the temperature scaling)
        kd_loss = tf.reduce_mean(kl_div_loss) * (self.temperature ** 2)
        
        return kd_loss

    
    def calculate_rl_loss(self, states, actions, old_log_probs, advantages, student_network):
        """
        Calculates the standard DRL loss (e.g., the PPO clipped loss) using the 
        Student's experience (States, Advantages). This provides 'on-the-job' learning.
        """
        # 1. Forward pass through the Student Network
        # new_logits, new_values = student_network(states)
        # new_log_probs = ... (calculate from new_logits)
        
        # 2. Calculate PPO Ratio and Clipped Surrogate Loss
        # ratio = tf.exp(new_log_probs - old_log_probs)
        # pg_loss1 = ratio * advantages
        # pg_loss2 = tf.clip_by_value(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * advantages
        
        # 3. Final PPO Loss
        rl_loss = -tf.reduce_mean(tf.minimum(pg_loss1, pg_loss2))
        
        return rl_loss

    
    def generate_hybrid_loss(self, student_network, teacher_logits, rl_inputs):
        """
        The main method: Generates the combined, weighted loss for the Student.
        
        Total_Loss = (Weight_KD * KD_Loss) + (Weight_RL * RL_Loss) + (Value_Loss + Entropy_Loss)
        """
        # Unpack RL Inputs (states, actions, old_log_probs, advantages, returns, etc.)
        states, actions, old_log_probs, advantages, returns, per_weights = rl_inputs 

        # 1. Get Student outputs (logits and value)
        student_logits, student_values = student_network(states)
        
        # 2. Calculate KD Loss
        kd_loss = self.calculate_kd_loss(teacher_logits, student_logits)
        
        # 3. Calculate RL Policy Loss
        rl_policy_loss = self.calculate_rl_loss(states, actions, old_log_probs, advantages, student_network)
        
        # 4. Calculate RL Value Loss (standard squared error)
        value_loss = tf.reduce_mean(tf.square(student_values - returns) * 0.5) 

        # 5. Combine the losses
        policy_loss_term = (self.kd_weight * kd_loss) + (self.rl_weight * rl_policy_loss)
        
        # 6. Add regularization terms (Value Loss and Entropy Loss)
        total_hybrid_loss = (
            policy_loss_term + 
            (self.config['ppo']['value_coef'] * value_loss) + 
            (self.config['ppo']['entropy_coef'] * entropy_loss)
        )
        
        return total_hybrid_loss, {
            'kd_loss': kd_loss, 
            'rl_policy_loss': rl_policy_loss, 
            'value_loss': value_loss
        }