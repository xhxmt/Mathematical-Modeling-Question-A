# Smoke Interference Strategy Optimization

## Project Overview
This project implements a mathematical modeling solution for optimizing smoke interference grenade deployment strategies to protect targets from missile attacks. The solution covers multiple scenarios from single drone-single grenade to multi-drone-multi-grenade configurations.

## Problem Structure
- **Problem 1**: Fixed strategy obscuration duration calculation
- **Problem 2**: Single grenade optimal deployment strategy  
- **Problem 3**: Single drone multiple grenades strategy
- **Problem 4**: Multiple drones single grenade strategy
- **Problem 5**: Multiple drones multiple grenades vs multiple missiles

## Directory Structure
```
smoke_interference_strategy/
├── src/                   # Core implementation modules
├── results/              # Optimization results and analysis
├── data/                 # Input parameters and test cases
├── visualization/        # 3D trajectory visualization
├── tests/               # Unit tests and validation
└── README.md           # This file
```

## Key Features
- Physics-based trajectory modeling
- Geometric obscuration analysis
- Multi-objective optimization
- 3D visualization of trajectories
- Performance analysis and validation

## Requirements
- Python 3.8+
- NumPy, SciPy, Matplotlib
- Optimization libraries (DEAP, PySwarms)