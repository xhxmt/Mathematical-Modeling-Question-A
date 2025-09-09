"""
Advanced optimization module for Jupyter-based smoke strategy optimization.

This module provides comprehensive optimization algorithms including genetic algorithms,
differential evolution, particle swarm optimization, and multi-objective optimization
specifically designed for smoke interference strategy problems.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass
import pandas as pd
from abc import ABC, abstractmethod
import time
import warnings
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp

# Import physics core
from physics_core import *

@dataclass
class OptimizationConfig:
    """Configuration for optimization algorithms"""
    population_size: int = 50
    max_generations: int = 100
    mutation_rate: float = 0.1
    crossover_rate: float = 0.8
    tolerance: float = 1e-6
    max_stagnation: int = 20
    parallel_workers: int = 4
    random_seed: Optional[int] = None
    verbose: bool = True

class OptimizationProblem(ABC):
    """Abstract base class for optimization problems"""
    
    def __init__(self, name: str, bounds: List[Tuple[float, float]], 
                 constraints: Optional[List[Callable]] = None):
        self.name = name
        self.bounds = bounds
        self.constraints = constraints or []
        self.dimension = len(bounds)
        self.evaluation_count = 0
        self.best_solution = None
        self.best_fitness = float('-inf')
        
    @abstractmethod
    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate a solution and return fitness value"""
        pass
    
    def is_feasible(self, solution: np.ndarray) -> bool:
        """Check if solution satisfies all constraints"""
        # Check bounds
        for i, (lower, upper) in enumerate(self.bounds):
            if not (lower <= solution[i] <= upper):
                return False
        
        # Check custom constraints
        for constraint in self.constraints:
            if not constraint(solution):
                return False
                
        return True
    
    def clip_to_bounds(self, solution: np.ndarray) -> np.ndarray:
        """Clip solution to bounds"""
        clipped = solution.copy()
        for i, (lower, upper) in enumerate(self.bounds):
            clipped[i] = np.clip(clipped[i], lower, upper)
        return clipped

class SmokeProblem2(OptimizationProblem):
    """Problem 2: Single grenade optimal deployment strategy"""
    
    def __init__(self, scenario_template: Dict[str, Any], 
                 config: PhysicsConfig = PhysicsConfig()):
        # Parameters: [drop_time, detonation_delay]
        bounds = [(0.1, 10.0), (0.1, 8.0)]
        super().__init__("Problem 2: Single Grenade Optimization", bounds)
        
        self.scenario_template = scenario_template
        self.config = config
        self.sim_engine = InteractiveSimulationEngine(time_step=0.05, config=config)
        
    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate solution: [drop_time, detonation_delay]"""
        self.evaluation_count += 1
        
        if not self.is_feasible(solution):
            return -1000.0
            
        drop_time, detonation_delay = solution
        
        try:
            # Create scenario with current parameters
            scenario = self._create_scenario(drop_time, detonation_delay)
            
            # Run simulation
            results = self.sim_engine.simulate_scenario(scenario, max_time=50.0)
            
            # Extract total obscuration duration
            total_duration = sum(
                result['total_duration'] 
                for result in results['obscuration_results'].values()
            )
            
            # Update best solution
            if total_duration > self.best_fitness:
                self.best_fitness = total_duration
                self.best_solution = solution.copy()
            
            return total_duration
            
        except Exception as e:
            warnings.warn(f"Evaluation failed: {str(e)}")
            return -1000.0
    
    def _create_scenario(self, drop_time: float, detonation_delay: float) -> Dict[str, Any]:
        """Create scenario with given parameters"""
        scenario = self.scenario_template.copy()
        scenario['drones']['FY1']['grenades'][0]['drop_time'] = drop_time
        scenario['drones']['FY1']['grenades'][0]['detonation_delay'] = detonation_delay
        return scenario

class SmokeProblem3(OptimizationProblem):
    """Problem 3: Single drone multiple grenades coordination"""
    
    def __init__(self, scenario_template: Dict[str, Any], num_grenades: int = 3,
                 config: PhysicsConfig = PhysicsConfig()):
        # Parameters: [drop_time1, delay1, drop_time2, delay2, ...]
        bounds = []
        for i in range(num_grenades):
            bounds.extend([(0.1, 15.0), (0.1, 8.0)])  # drop_time, detonation_delay
            
        super().__init__("Problem 3: Multi-Grenade Coordination", bounds)
        
        self.scenario_template = scenario_template
        self.num_grenades = num_grenades
        self.config = config
        self.sim_engine = InteractiveSimulationEngine(time_step=0.05, config=config)
        
    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate solution: [drop_time1, delay1, drop_time2, delay2, ...]"""
        self.evaluation_count += 1
        
        if not self.is_feasible(solution):
            return -1000.0
        
        try:
            # Parse parameters
            grenades = []
            for i in range(self.num_grenades):
                drop_time = solution[2*i]
                detonation_delay = solution[2*i + 1]
                grenades.append({
                    'grenade_id': i + 1,
                    'drop_time': drop_time,
                    'detonation_delay': detonation_delay
                })
            
            # Create scenario
            scenario = self._create_scenario(grenades)
            
            # Run simulation
            results = self.sim_engine.simulate_scenario(scenario, max_time=80.0)
            
            # Calculate total obscuration
            total_duration = sum(
                result['total_duration'] 
                for result in results['obscuration_results'].values()
            )
            
            # Apply penalty for overlapping clouds (optional)
            penalty = self._calculate_overlap_penalty(grenades)
            fitness = total_duration - penalty
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()
            
            return fitness
            
        except Exception as e:
            warnings.warn(f"Evaluation failed: {str(e)}")
            return -1000.0
    
    def _create_scenario(self, grenades: List[Dict]) -> Dict[str, Any]:
        """Create scenario with multiple grenades"""
        scenario = self.scenario_template.copy()
        scenario['drones']['FY1']['grenades'] = grenades
        return scenario
    
    def _calculate_overlap_penalty(self, grenades: List[Dict]) -> float:
        """Calculate penalty for overlapping cloud periods"""
        penalty = 0.0
        
        for i in range(len(grenades)):
            for j in range(i + 1, len(grenades)):
                det_time1 = grenades[i]['drop_time'] + grenades[i]['detonation_delay']
                det_time2 = grenades[j]['drop_time'] + grenades[j]['detonation_delay']
                fade_time1 = det_time1 + self.config.CLOUD_LIFETIME
                fade_time2 = det_time2 + self.config.CLOUD_LIFETIME
                
                # Calculate overlap duration
                overlap_start = max(det_time1, det_time2)
                overlap_end = min(fade_time1, fade_time2)
                
                if overlap_start < overlap_end:
                    overlap_duration = overlap_end - overlap_start
                    penalty += overlap_duration * 0.1  # Small penalty weight
                    
        return penalty

class SmokeProblem5(OptimizationProblem):
    """Problem 5: Multi-drone multi-grenade vs multi-missile comprehensive strategy"""
    
    def __init__(self, scenario_template: Dict[str, Any], 
                 config: PhysicsConfig = PhysicsConfig()):
        # Parameters: [drone1_x, drone1_y, drone1_vx, drone1_vy, drop_time1, delay1, 
        #              drone2_x, drone2_y, drone2_vx, drone2_vy, drop_time2, delay2]
        bounds = [
            # Drone FY1 position and velocity
            (15000, 19000), (-1000, 1000),  # initial position x, y
            (-150, -80), (-50, 50),         # velocity x, y
            (0.1, 10.0), (0.1, 8.0),       # drop_time, detonation_delay
            
            # Drone FY2 position and velocity  
            (15000, 19000), (-1000, 1000),  # initial position x, y
            (-150, -80), (-50, 50),         # velocity x, y  
            (0.1, 10.0), (0.1, 8.0),       # drop_time, detonation_delay
        ]
        
        super().__init__("Problem 5: Multi-Drone Multi-Missile Strategy", bounds)
        
        self.scenario_template = scenario_template
        self.config = config
        self.sim_engine = InteractiveSimulationEngine(time_step=0.05, config=config)
        
    def evaluate(self, solution: np.ndarray) -> float:
        """Evaluate comprehensive strategy solution"""
        self.evaluation_count += 1
        
        if not self.is_feasible(solution):
            return -1000.0
            
        try:
            # Parse parameters
            fy1_x, fy1_y, fy1_vx, fy1_vy, drop1, delay1 = solution[:6]
            fy2_x, fy2_y, fy2_vx, fy2_vy, drop2, delay2 = solution[6:]
            
            # Create scenario
            scenario = self._create_scenario(
                fy1_x, fy1_y, fy1_vx, fy1_vy, drop1, delay1,
                fy2_x, fy2_y, fy2_vx, fy2_vy, drop2, delay2
            )
            
            # Run simulation
            results = self.sim_engine.simulate_scenario(scenario, max_time=100.0)
            
            # Calculate total obscuration across all missiles
            total_duration = sum(
                result['total_duration'] 
                for result in results['obscuration_results'].values()
            )
            
            # Apply coordination bonus
            coordination_bonus = self._calculate_coordination_bonus(results)
            fitness = total_duration + coordination_bonus
            
            if fitness > self.best_fitness:
                self.best_fitness = fitness
                self.best_solution = solution.copy()
            
            return fitness
            
        except Exception as e:
            warnings.warn(f"Evaluation failed: {str(e)}")
            return -1000.0
    
    def _create_scenario(self, fy1_x, fy1_y, fy1_vx, fy1_vy, drop1, delay1,
                        fy2_x, fy2_y, fy2_vx, fy2_vy, drop2, delay2) -> Dict[str, Any]:
        """Create multi-drone scenario"""
        scenario = {
            'missiles': self.scenario_template['missiles'].copy(),
            'drones': {
                'FY1': {
                    'initial_pos': np.array([fy1_x, fy1_y, 1800.0]),
                    'velocity': np.array([fy1_vx, fy1_vy, 0.0]),
                    'grenades': [{'grenade_id': 1, 'drop_time': drop1, 'detonation_delay': delay1}]
                },
                'FY2': {
                    'initial_pos': np.array([fy2_x, fy2_y, 1800.0]),
                    'velocity': np.array([fy2_vx, fy2_vy, 0.0]),
                    'grenades': [{'grenade_id': 1, 'drop_time': drop2, 'detonation_delay': delay2}]
                }
            },
            'target_points': self.scenario_template['target_points']
        }
        
        return scenario
    
    def _calculate_coordination_bonus(self, results) -> float:
        """Calculate bonus for good coordination between drones"""
        if len(results['cloud_data']) < 2:
            return 0.0
            
        # Simple coordination metric: reward non-overlapping but complementary timing
        clouds = results['cloud_data']
        cloud1, cloud2 = clouds[0], clouds[1]
        
        det1 = cloud1['detonation_time']
        det2 = cloud2['detonation_time']
        fade1 = cloud1['fade_time']
        fade2 = cloud2['fade_time']
        
        # Bonus for sequential coverage
        if fade1 <= det2 or fade2 <= det1:
            # No overlap - good sequential coverage
            gap = min(abs(det2 - fade1), abs(det1 - fade2))
            if gap < 5.0:  # Small gap is good
                return 0.5  # Small bonus
                
        return 0.0

class DifferentialEvolution:
    """Differential Evolution optimizer for smoke strategy problems"""
    
    def __init__(self, config: OptimizationConfig = OptimizationConfig()):
        self.config = config
        self.history = []
        
    def optimize(self, problem: OptimizationProblem) -> Tuple[np.ndarray, float, Dict]:
        """Run differential evolution optimization"""
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            
        # Initialize population
        population = self._initialize_population(problem)
        fitness_values = np.array([problem.evaluate(ind) for ind in population])
        
        best_idx = np.argmax(fitness_values)
        best_individual = population[best_idx].copy()
        best_fitness = fitness_values[best_idx]
        
        stagnation_count = 0
        
        if self.config.verbose:
            print(f"🚀 Starting Differential Evolution for {problem.name}")
            print(f"Population size: {self.config.population_size}")
            print(f"Dimensions: {problem.dimension}")
            print(f"Initial best fitness: {best_fitness:.6f}")
            
        for generation in range(self.config.max_generations):
            new_population = []
            new_fitness = []
            
            for i in range(self.config.population_size):
                # Mutation
                mutant = self._mutate(population, i)
                mutant = problem.clip_to_bounds(mutant)
                
                # Crossover
                trial = self._crossover(population[i], mutant)
                
                # Selection
                trial_fitness = problem.evaluate(trial)
                
                if trial_fitness > fitness_values[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness_values[i])
                    
                # Update global best
                if trial_fitness > best_fitness:
                    best_individual = trial.copy()
                    best_fitness = trial_fitness
                    stagnation_count = 0
                    
            population = new_population
            fitness_values = np.array(new_fitness)
            
            # Record history
            generation_stats = {
                'generation': generation,
                'best_fitness': best_fitness,
                'mean_fitness': np.mean(fitness_values),
                'std_fitness': np.std(fitness_values),
                'evaluations': problem.evaluation_count
            }
            self.history.append(generation_stats)
            
            if self.config.verbose and (generation + 1) % 10 == 0:
                print(f"Gen {generation+1}: Best={best_fitness:.6f}, "
                      f"Mean={np.mean(fitness_values):.6f}, "
                      f"Std={np.std(fitness_values):.6f}")
                      
            # Check termination conditions
            stagnation_count += 1
            if stagnation_count >= self.config.max_stagnation:
                if self.config.verbose:
                    print(f"Early termination due to stagnation at generation {generation+1}")
                break
                
            if np.std(fitness_values) < self.config.tolerance:
                if self.config.verbose:
                    print(f"Convergence achieved at generation {generation+1}")
                break
        
        # Final results
        final_stats = {
            'total_generations': generation + 1,
            'total_evaluations': problem.evaluation_count,
            'final_population_std': np.std(fitness_values),
            'convergence_history': self.history
        }
        
        if self.config.verbose:
            print(f"✅ Optimization completed!")
            print(f"Best fitness: {best_fitness:.6f}")
            print(f"Total evaluations: {problem.evaluation_count}")
            print(f"Final solution: {best_individual}")
            
        return best_individual, best_fitness, final_stats
    
    def _initialize_population(self, problem: OptimizationProblem) -> List[np.ndarray]:
        """Initialize random population within bounds"""
        population = []
        for _ in range(self.config.population_size):
            individual = np.array([
                np.random.uniform(lower, upper)
                for lower, upper in problem.bounds
            ])
            population.append(individual)
        return population
    
    def _mutate(self, population: List[np.ndarray], current_idx: int) -> np.ndarray:
        """DE/rand/1 mutation strategy"""
        indices = list(range(len(population)))
        indices.remove(current_idx)
        
        # Select three random different individuals
        selected = np.random.choice(indices, 3, replace=False)
        r1, r2, r3 = [population[i] for i in selected]
        
        # Mutation: mutant = r1 + F * (r2 - r3)
        F = 0.5  # Mutation factor
        mutant = r1 + F * (r2 - r3)
        
        return mutant
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Binomial crossover"""
        trial = target.copy()
        
        # Ensure at least one parameter comes from mutant
        j_rand = np.random.randint(len(target))
        
        for j in range(len(target)):
            if np.random.random() < self.config.crossover_rate or j == j_rand:
                trial[j] = mutant[j]
                
        return trial

class ParticleSwarmOptimization:
    """Particle Swarm Optimization for smoke strategy problems"""
    
    def __init__(self, config: OptimizationConfig = OptimizationConfig()):
        self.config = config
        self.history = []
        
    def optimize(self, problem: OptimizationProblem) -> Tuple[np.ndarray, float, Dict]:
        """Run PSO optimization"""
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)
            
        # PSO parameters
        w = 0.9  # Inertia weight
        c1 = 2.0  # Cognitive parameter
        c2 = 2.0  # Social parameter
        
        # Initialize particles
        particles = self._initialize_particles(problem)
        velocities = [np.zeros(problem.dimension) for _ in range(self.config.population_size)]
        
        # Evaluate initial positions
        fitness_values = [problem.evaluate(particle) for particle in particles]
        
        # Initialize personal bests
        personal_bests = [particle.copy() for particle in particles]
        personal_best_fitness = fitness_values.copy()
        
        # Initialize global best
        best_idx = np.argmax(fitness_values)
        global_best = particles[best_idx].copy()
        global_best_fitness = fitness_values[best_idx]
        
        if self.config.verbose:
            print(f"🚀 Starting Particle Swarm Optimization for {problem.name}")
            print(f"Swarm size: {self.config.population_size}")
            print(f"Dimensions: {problem.dimension}")
            print(f"Initial best fitness: {global_best_fitness:.6f}")
            
        stagnation_count = 0
        
        for iteration in range(self.config.max_generations):
            for i in range(self.config.population_size):
                # Update velocity
                r1, r2 = np.random.random(problem.dimension), np.random.random(problem.dimension)
                
                velocities[i] = (w * velocities[i] + 
                               c1 * r1 * (personal_bests[i] - particles[i]) +
                               c2 * r2 * (global_best - particles[i]))
                
                # Update position
                particles[i] = particles[i] + velocities[i]
                particles[i] = problem.clip_to_bounds(particles[i])
                
                # Evaluate new position
                fitness = problem.evaluate(particles[i])
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_bests[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                # Update global best
                if fitness > global_best_fitness:
                    global_best = particles[i].copy()
                    global_best_fitness = fitness
                    stagnation_count = 0
                    
            # Record history
            current_fitness = [problem.evaluate(particle) for particle in particles]
            iteration_stats = {
                'iteration': iteration,
                'best_fitness': global_best_fitness,
                'mean_fitness': np.mean(current_fitness),
                'std_fitness': np.std(current_fitness),
                'evaluations': problem.evaluation_count
            }
            self.history.append(iteration_stats)
            
            if self.config.verbose and (iteration + 1) % 10 == 0:
                print(f"Iter {iteration+1}: Best={global_best_fitness:.6f}, "
                      f"Mean={np.mean(current_fitness):.6f}")
                      
            # Check termination
            stagnation_count += 1
            if stagnation_count >= self.config.max_stagnation:
                if self.config.verbose:
                    print(f"Early termination due to stagnation at iteration {iteration+1}")
                break
        
        final_stats = {
            'total_iterations': iteration + 1,
            'total_evaluations': problem.evaluation_count,
            'convergence_history': self.history
        }
        
        if self.config.verbose:
            print(f"✅ PSO completed!")
            print(f"Best fitness: {global_best_fitness:.6f}")
            print(f"Total evaluations: {problem.evaluation_count}")
            
        return global_best, global_best_fitness, final_stats
    
    def _initialize_particles(self, problem: OptimizationProblem) -> List[np.ndarray]:
        """Initialize particle swarm within bounds"""
        particles = []
        for _ in range(self.config.population_size):
            particle = np.array([
                np.random.uniform(lower, upper)
                for lower, upper in problem.bounds
            ])
            particles.append(particle)
        return particles

class MultiObjectiveOptimizer:
    """Multi-objective optimization for smoke strategy problems"""
    
    def __init__(self, config: OptimizationConfig = OptimizationConfig()):
        self.config = config
        self.pareto_front = []
        self.history = []
        
    def optimize_pareto(self, problem: OptimizationProblem, 
                       secondary_objectives: List[Callable]) -> Tuple[List[np.ndarray], List[Tuple], Dict]:
        """Find Pareto-optimal solutions"""
        # This is a simplified multi-objective approach
        # In practice, you'd use NSGA-II, MOEA/D, etc.
        
        if self.config.verbose:
            print(f"🎯 Starting Multi-Objective Optimization for {problem.name}")
            print(f"Number of objectives: {len(secondary_objectives) + 1}")
            
        # Generate diverse solutions
        solutions = []
        objectives = []
        
        for _ in range(self.config.population_size * 2):  # Generate more solutions
            # Random solution
            solution = np.array([
                np.random.uniform(lower, upper)
                for lower, upper in problem.bounds
            ])
            
            if not problem.is_feasible(solution):
                continue
                
            # Evaluate all objectives
            primary_obj = problem.evaluate(solution)
            secondary_objs = [obj(solution, problem) for obj in secondary_objectives]
            
            solutions.append(solution)
            objectives.append((primary_obj,) + tuple(secondary_objs))
        
        # Find Pareto front
        pareto_solutions, pareto_objectives = self._find_pareto_front(solutions, objectives)
        
        final_stats = {
            'total_solutions_evaluated': len(solutions),
            'pareto_front_size': len(pareto_solutions),
            'total_evaluations': problem.evaluation_count
        }
        
        if self.config.verbose:
            print(f"✅ Multi-objective optimization completed!")
            print(f"Pareto front size: {len(pareto_solutions)}")
            print(f"Total evaluations: {problem.evaluation_count}")
            
        return pareto_solutions, pareto_objectives, final_stats
    
    def _find_pareto_front(self, solutions: List[np.ndarray], 
                          objectives: List[Tuple]) -> Tuple[List[np.ndarray], List[Tuple]]:
        """Find Pareto-optimal solutions using simple dominance sorting"""
        pareto_solutions = []
        pareto_objectives = []
        
        for i, (sol1, obj1) in enumerate(zip(solutions, objectives)):
            is_dominated = False
            
            for j, (sol2, obj2) in enumerate(zip(solutions, objectives)):
                if i == j:
                    continue
                    
                # Check if obj1 is dominated by obj2
                if all(o2 >= o1 for o1, o2 in zip(obj1, obj2)) and any(o2 > o1 for o1, o2 in zip(obj1, obj2)):
                    is_dominated = True
                    break
            
            if not is_dominated:
                pareto_solutions.append(sol1)
                pareto_objectives.append(obj1)
                
        return pareto_solutions, pareto_objectives

# Utility functions
def create_default_problem2_scenario() -> Dict[str, Any]:
    """Create default scenario for Problem 2"""
    return {
        'missiles': {
            'M1': {
                'initial_pos': np.array([20000.0, 0.0, 2000.0]),
                'target_pos': np.array([0.0, 200.0, 5.0]),  # Real target
                'speed': 300.0
            }
        },
        'drones': {
            'FY1': {
                'initial_pos': np.array([17800.0, 0.0, 1800.0]),
                'velocity': np.array([-120.0, 0.0, 0.0]),
                'grenades': [
                    {'grenade_id': 1, 'drop_time': 1.5, 'detonation_delay': 3.6}
                ]
            }
        },
        'target_points': [
            np.array([0.0, 200.0, 10.0]),   # Top
            np.array([0.0, 200.0, 0.0]),    # Bottom
            np.array([200.0, 200.0, 5.0]),  # Side
            np.array([-200.0, 200.0, 5.0])  # Side
        ]
    }

def create_default_problem5_scenario() -> Dict[str, Any]:
    """Create default scenario for Problem 5 (multi-missile)"""
    return {
        'missiles': {
            'M1': {
                'initial_pos': np.array([20000.0, 0.0, 2000.0]),
                'target_pos': np.array([0.0, 200.0, 5.0]),  # Real target
                'speed': 300.0
            },
            'M2': {
                'initial_pos': np.array([18000.0, 2000.0, 1800.0]),
                'target_pos': np.array([0.0, 200.0, 5.0]),  # Real target
                'speed': 280.0
            }
        },
        'drones': {
            'FY1': {
                'initial_pos': np.array([17800.0, 0.0, 1800.0]),
                'velocity': np.array([-120.0, 0.0, 0.0]),
                'grenades': [
                    {'grenade_id': 1, 'drop_time': 1.5, 'detonation_delay': 3.6}
                ]
            }
        },
        'target_points': [
            np.array([0.0, 200.0, 10.0]),
            np.array([0.0, 200.0, 0.0]),
            np.array([200.0, 200.0, 5.0]),
            np.array([-200.0, 200.0, 5.0])
        ]
    }

# Secondary objective functions for multi-objective optimization
def minimize_resource_usage(solution: np.ndarray, problem: OptimizationProblem) -> float:
    """Secondary objective: minimize resource usage (number of grenades, flight time, etc.)"""
    if isinstance(problem, SmokeProblem3):
        # For multi-grenade problems, penalize total deployment time
        total_deployment_time = 0
        for i in range(problem.num_grenades):
            drop_time = solution[2*i]
            detonation_delay = solution[2*i + 1]
            total_deployment_time += drop_time + detonation_delay
        return -total_deployment_time  # Negative because we want to minimize
    
    elif isinstance(problem, SmokeProblem5):
        # For multi-drone problems, penalize total flight distance
        fy1_vx, fy1_vy = solution[2:4]
        fy2_vx, fy2_vy = solution[8:10]
        
        total_speed = np.sqrt(fy1_vx**2 + fy1_vy**2) + np.sqrt(fy2_vx**2 + fy2_vy**2)
        return -total_speed  # Negative because we want to minimize
    
    return 0.0

def maximize_robustness(solution: np.ndarray, problem: OptimizationProblem) -> float:
    """Secondary objective: maximize solution robustness against parameter uncertainties"""
    # Simplified robustness measure: evaluate solution with small perturbations
    base_fitness = problem.evaluate(solution)
    
    perturbation_count = 5
    perturbed_fitness = []
    
    for _ in range(perturbation_count):
        # Add small random perturbation
        noise_scale = 0.05  # 5% noise
        perturbed = solution + np.random.normal(0, noise_scale, len(solution))
        perturbed = problem.clip_to_bounds(perturbed)
        
        fitness = problem.evaluate(perturbed)
        perturbed_fitness.append(fitness)
    
    # Robustness = negative standard deviation of perturbed fitness
    robustness = -np.std(perturbed_fitness)
    
    return robustness

# Export main classes and functions
__all__ = [
    'OptimizationConfig',
    'OptimizationProblem', 
    'SmokeProblem2',
    'SmokeProblem3',
    'SmokeProblem5',
    'DifferentialEvolution',
    'ParticleSwarmOptimization', 
    'MultiObjectiveOptimizer',
    'create_default_problem2_scenario',
    'create_default_problem5_scenario',
    'minimize_resource_usage',
    'maximize_robustness'
]