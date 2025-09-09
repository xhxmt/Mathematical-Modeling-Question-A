"""
Jupyter Smoke Strategy Package

A comprehensive Jupyter-based solution for smoke interference strategy optimization.
"""

__version__ = "2.0.0"
__author__ = "Mathematical Modeling Team - Jupyter Edition"

# Core imports
try:
    from .physics_core import *
except ImportError as e:
    print(f"Warning: Could not import physics_core: {e}")

try:
    from .optimization import *
except ImportError as e:
    print(f"Warning: Could not import optimization: {e}")

try:
    from .visualization import *
except ImportError as e:
    print(f"Warning: Could not import visualization: {e}")

try:
    from .utils import *
except ImportError as e:
    print(f"Warning: Could not import utils: {e}")