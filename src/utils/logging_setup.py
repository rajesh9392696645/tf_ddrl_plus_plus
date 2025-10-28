# src/utils/logging_setup.py

import os
import datetime
import tensorflow as tf
# Import the hparams API for hyperparameter logging in TensorBoard
from tensorboard.plugins.hparams import api as hp 
import sys
# If needed, adjust the relative import path based on your exact structure
from ..utils.config_loader import ConfigLoader 

# Optional: Import Weights & Biases if used for advanced tracking
# import wandb 

class LoggingSetup:
    """
    Configures and initializes logging tools (TensorBoard, W&B) for 
    tracking experiment metrics and hyperparameters.
    """

    def __init__(self, config_path):
        """
        Initializes the loader by finding and reading the configuration file,
        and setting up the log directories.
        """
        self.config = ConfigLoader(config_path).get_config()
        self.log_dir_base = self.config['logging']['log_dir_base']
        self.experiment_name = self.config['logging']['experiment_name']
        self.use_wandb = self.config['logging'].get('use_wandb', False)
        
        # Create a unique run identifier
        self.run_id = self._create_run_id()
        # The full log path is unique for each run
        self.full_log_path = os.path.join(self.log_dir_base, self.run_id)
        
        # Initialize logging writers
        self.tb_writer = self._setup_tensorboard()
        if self.use_wandb:
            self._setup_wandb()
            
    def _create_run_id(self):
        """Generates a unique directory name for the current experiment run."""
        timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        return f"{self.experiment_name}_{timestamp}"

    def _setup_tensorboard(self):
        """Initializes the TensorFlow SummaryWriter for TensorBoard logging."""
        print(f"Setting up TensorBoard logging at: {self.full_log_path}")
        # Create the log directory
        os.makedirs(self.full_log_path, exist_ok=True)
        # Return the TensorFlow file writer instance
        return tf.summary.create_file_writer(self.full_log_path)

    def _setup_wandb(self):
        """Initializes Weights & Biases tracking."""
        print("Setting up Weights & Biases logging.")
        # wandb.init(...)
        pass

    def log_scalar(self, step, tag, value, component='tb'):
        """
        Writes a single scalar value (e.g., loss, reward) to the logging system.
        """
        if component == 'tb':
            with self.tb_writer.as_default():
                tf.summary.scalar(tag, value, step=step)
            # Ensure data is written immediately to disk
            self.tb_writer.flush()
            
        # if self.use_wandb and component == 'wandb':
        #     wandb.log({tag: value}, step=step)

    def log_hparams(self):
        """Logs the entire set of hyperparameters, using the hp API."""
    
        # Flatten the nested config dictionary into a single dictionary
        flat_hparams = {}
        for section, settings in self.config.items():
            for key, value in settings.items():
                # Convert values to strings or basic types
                flat_hparams[f"{section}/{key}"] = str(value) 

        # Define dummy metrics for the HParams dashboard to display
        metric_reward = hp.Metric('metrics/avg_reward', display_name='Avg Reward')
        metric_makespan = hp.Metric('metrics/avg_makespan', display_name='Avg Makespan')
    
        # Log the hparams config for this run
        with self.tb_writer.as_default():
        # 1. Log the configuration for this run
            hp.hparams(flat_hparams)
        
        # 2. Log initial dummy values for the hparams dashboard to populate
        # FIX: Use the private attribute ._tag (for compatibility with your TF/TB version)
        tf.summary.scalar(metric_reward._tag, 0.0, step=0) 
        tf.summary.scalar(metric_makespan._tag, 9999.0, step=0)
        
    print("Hyperparameters logged to TensorBoard HParams dashboard.")

# Example usage (for reference, not executed during import)
# if __name__ == '__main__':
#     # ... loader = ConfigLoader('config.yaml')
#     # ... logger = LoggingSetup('config.yaml')
#     # ... logger.log_hparams()