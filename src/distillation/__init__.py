# src/distillation/__init__.py

# Import the main classes to make them directly accessible from the 'distillation' package.
# This simplifies imports in the main scripts (like main_edge_deployer.py)
# e.g., 'from src.distillation import KnowledgeDistiller'

from .knowledge_distiller import KnowledgeDistiller # Logic for calculating KD loss
from .hybrid_feedback import HybridFeedbackGenerator # Assuming a class for generating gradient correction signals

# Use the '__all__' list to explicitly define what names are exported
__all__ = [
    "KnowledgeDistiller",
    "HybridFeedbackGenerator", 
]

# The rest of the file would typically be empty.