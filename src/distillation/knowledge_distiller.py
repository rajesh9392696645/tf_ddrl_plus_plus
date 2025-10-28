# src/distillation/knowledge_distiller.py

import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Import necessary components
from ..arch.teacher_network import TeacherNetwork
from ..arch.student_network import StudentNetwork
from ..core.per_buffer import PrioritizedExperienceReplayBuffer 
from .hybrid_feedback import HybridFeedbackGenerator
from ..utils.config_loader import ConfigLoader

class KnowledgeDistiller:
    """
    Manages the Knowledge Distillation process to train the Student Network.
    It orchestrates the data flow, Teacher inference, and hybrid loss calculation.
    """
    def __init__(self, config_path, teacher_model_path, env_params):
        self.config = ConfigLoader(config_path).get_config()
        self.optimizer = Adam(learning_rate=self.config['distillation']['student_lr'])
        
        # 1. Initialize Teacher and Student Models
        # The Teacher is loaded (and frozen) for inference only
        self.teacher = TeacherNetwork(**self.config['arch'])
        self.teacher.load_weights(teacher_model_path).expect_partial()
        
        self.student = StudentNetwork(
            action_space_size=env_params['action_size'], 
            state_feature_dim=env_params['feature_dim'],
            **self.config['arch']
        )
        
        # 2. Initialize Training Components
        # The Student often uses the same replay buffer used for the Teacher's training data.
        self.buffer = PrioritizedExperienceReplayBuffer(**self.config['per']) 
        self.loss_generator = HybridFeedbackGenerator(config_path)
        
        # Hyperparameters
        self.kd_epochs = self.config['distillation']['kd_epochs']
        self.batch_size = self.config['distillation']['batch_size']
        

    def train_step(self, rl_inputs):
        """
        Performs a single, integrated training step for the Student Network.
        This step is responsible for generating the Teacher's targets and applying the hybrid loss.
        """
        # Unpack Inputs
        states, actions, old_log_probs, advantages, returns, per_weights = rl_inputs 
        
        # --- Teacher Inference (Target Generation) ---
        # The Teacher provides the 'expert' targets for the Student's policy.
        # Note: Teacher is not included in the tape, as it is fixed/frozen.
        teacher_logits, _ = self.teacher(states, training=False)
        
        with tf.GradientTape() as tape:
            # Generate the blended loss (KD + RL)
            total_loss, loss_details = self.loss_generator.generate_hybrid_loss(
                student_network=self.student,
                teacher_logits=teacher_logits,
                rl_inputs=rl_inputs
            )

        # Apply Gradients
        gradients = tape.gradient(total_loss, self.student.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.student.trainable_variables))
        
        return total_loss, loss_details

    
    def run_distillation(self, total_training_steps):
        """
        The main loop for the Knowledge Distillation training phase.
        The Student learns for multiple epochs over sampled data from the buffer.
        """
        for step in range(total_training_steps):
            # 1. Anneal the PER beta parameter
            self.buffer.anneal_beta(step, total_training_steps)

            # 2. Sample a batch of data from the Replay Buffer
            # This data (states, actions, returns, etc.) comes from the Teacher's prior experience.
            states, actions, old_log_probs, advantages, returns, per_indices, per_weights = self.buffer.sample(self.batch_size)
            
            # 3. Package the RL inputs
            rl_inputs = (states, actions, old_log_probs, advantages, returns, per_weights)

            # 4. Perform the KD/RL training update
            loss, details = self.train_step(rl_inputs)

            # 5. Optional: Update PER priorities (If the hybrid loss generates a TD-Error update)
            # The structure suggests the TD-error update might be managed by PPO or the loss generator.
            
            # 6. Log and Save
            if step % self.config['logging']['save_interval'] == 0:
                self.student.save_weights(f'student_checkpoint_{step}')