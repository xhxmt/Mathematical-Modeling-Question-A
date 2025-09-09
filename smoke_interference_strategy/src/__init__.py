"""
Smoke Interference Strategy Optimization Package

A comprehensive mathematical modeling solution for optimizing smoke interference 
grenade deployment strategies to protect targets from missile attacks.
"""

__version__ = "1.0.0"
__author__ = "Mathematical Modeling Team"

from .physics_core import (
    PhysicsConstants, 
    Vector3D, 
    TrajectoryCalculator, 
    ObscurationAnalyzer, 
    SimulationEngine
)

from .problem1_solver import Problem1Solver
from .problem2_solver import Problem2Solver  
from .problem3_solver import Problem3Solver
from .problem4_solver import Problem4Solver
from .problem5_solver import Problem5Solver

# Visualization imports are optional due to matplotlib dependency issues
try:
    from .visualization import Visualization3D, create_scenario_data_from_results, save_visualization_plots
    _visualization_available = True
except ImportError:
    _visualization_available = False

__all__ = [
    'PhysicsConstants',
    'Vector3D', 
    'TrajectoryCalculator',
    'ObscurationAnalyzer',
    'SimulationEngine',
    'Problem1Solver',
    'Problem2Solver', 
    'Problem3Solver',
    'Problem4Solver',
    'Problem5Solver'
]

if _visualization_available:
    __all__.extend(['Visualization3D', 'create_scenario_data_from_results', 'save_visualization_plots'])