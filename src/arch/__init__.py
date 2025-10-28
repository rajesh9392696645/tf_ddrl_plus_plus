# src/arch/__init__.py

# Import key classes to make them directly accessible from 'arch'
# This allows for cleaner imports elsewhere, e.g., 
# 'from src.arch import TeacherNetwork' instead of 'from src.arch.teacher_network import TeacherNetwork'

from .graph_transformer import SparseGraphTransformerEncoder
from .teacher_network import TeacherNetwork  # Assuming the main class is TeacherNetwork
from .student_network import StudentNetwork  # Assuming the main class is StudentNetwork

# You can also use the '__all__' list to explicitly define what is exported
__all__ = [
    "SparseGraphTransformerEncoder",
    "TeacherNetwork",
    "StudentNetwork",
]

# The rest of the file would typically be empty or contain simple package-level configuration.