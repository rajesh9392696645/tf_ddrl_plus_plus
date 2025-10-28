# src/utils/config_loader.py

import yaml
import os
import sys

class ConfigLoader:
    """
    Handles loading, parsing, and validating the 'config.yaml' file.
    Ensures hyperparameters and settings are available throughout the project.
    """

    def __init__(self, config_path='config.yaml'):
        """
        Initializes the loader by finding and reading the configuration file.
        """
        self.config_path = config_path
        self._config_data = self._load_config()
        self._validate_config() # Optional but good practice

    def _load_config(self):
        """
        Opens and loads the YAML file content.
        """
        if not os.path.exists(self.config_path):
            print(f"Error: Configuration file not found at {self.config_path}")
            sys.exit(1)
            
        try:
            with open(self.config_path, 'r') as f:
                # Use FullLoader for security and reliability with recent PyYAML versions
                config_data = yaml.load(f, Loader=yaml.FullLoader)
            return config_data
        except yaml.YAMLError as exc:
            print(f"Error parsing YAML file: {exc}")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred while loading config: {e}")
            sys.exit(1)

    def _validate_config(self):
        """
        Checks for the presence of mandatory top-level keys.
        (e.g., 'drl', 'arch', 'env', 'ppo', 'distillation').
        """
        required_keys = ['drl', 'arch', 'env', 'ppo', 'distillation', 'logging']
        
        for key in required_keys:
            if key not in self._config_data:
                raise ValueError(f"Configuration missing required section: '{key}'. Check config.yaml.")
                
        # Further checks could ensure learning rates are positive, etc.
        # if self._config_data['drl']['learning_rate'] <= 0:
        #     raise ValueError("DRL learning rate must be positive.")
        pass

    def get_config(self):
        """
        Returns the entire loaded configuration dictionary.
        """
        return self._config_data

    def get_section(self, section_name):
        """
        Returns a specific section (e.g., 'ppo' or 'arch') of the configuration.
        """
        if section_name in self._config_data:
            return self._config_data[section_name]
        else:
            raise KeyError(f"Configuration section '{section_name}' not found.")

# Example usage (for internal testing/demonstration)
if __name__ == '__main__':
    # This block would only run if the file is executed directly.
    # In a real setup, it would load the config.yaml located in the project root.
    try:
        loader = ConfigLoader('../../config.yaml') # Adjust path if needed for testing
        drl_settings = loader.get_section('drl')
        print("Successfully loaded DRL Settings:")
        print(drl_settings)
    except Exception as e:
        print(f"Test failed: {e}")