# src/core/__init__.py

# Import key classes to make them directly accessible from the 'core' package.
# This simplifies imports elsewhere, e.g., 
# 'from src.core import PPOAgent' instead of 'from src.core.ppo_agent import PPOAgent'

from .ppo_agent import PPOAgent
from .per_buffer import PrioritizedExperienceReplayBuffer # Assuming the main class is PrioritizedExperienceReplayBuffer

# Use the '__all__' list to explicitly define what names are exported
__all__ = [
    "PPOAgent",
    "PrioritizedExperienceReplayBuffer",
]

# The rest of the file would typically be empty.