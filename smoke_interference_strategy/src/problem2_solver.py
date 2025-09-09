"""
Problem 2: Optimal deployment strategy for single drone and single grenade.

This module implements optimization algorithms to find the best drone flight strategy
and grenade deployment timing to maximize obscuration duration against missile M1.
"""

import numpy as np
from typing import List, Dict, Any, Tuple, Callable
from scipy.optimize import differential_evolution, minimize
import warnings
from .physics_core import (
    PhysicsConstants, TrajectoryCalculator, ObscurationAnalyzer, 
    SimulationEngine, Vector3D
)

class Problem2Solver:
    """Solver for Problem 2: Single Grenade Optimal Deployment Strategy"""
    
    def __init__(self, simulation_time_step: float = 0.02):
        """
        Initialize Problem 2 solver with optimization capabilities
        
        Args:
            simulation_time_step: Time step for numerical simulation
        """
        self.simulation_engine = SimulationEngine(simulation_time_step)
        
        # Fixed parameters
        self.missile_initial_pos = np.array([20000.0, 0.0, 2000.0])  # M1 initial position
        self.drone_initial_pos = np.array([17800.0, 0.0, 1800.0])   # FY1 initial position  
        self.fake_target_pos = np.array([0.0, 0.0, 0.0])            # Fake target at origin
        
        # Real target definition
        self.real_target_center = np.array([0.0, 200.0, 5.0])
        self.real_target_radius = 200.0
        self.real_target_height = 10.0
        self.target_key_points = self._generate_target_key_points()
        
        # Decision variable bounds
        self.bounds = {
            'flight_angle': (0.0, 2*np.pi),           # Drone flight direction [rad]
            'flight_speed': (70.0, 140.0),            # Drone speed [m/s]
            'drop_time': (0.1, 60.0),                 # Drop time [s]
            'detonation_delay': (0.1, 10.0)           # Detonation delay [s]
        }
        
        # Optimization tracking
        self.evaluation_count = 0
        self.best_solutions = []
        
    def _generate_target_key_points(self) -> List[np.ndarray]:
        """Generate key points on the cylindrical real target"""
        points = []
        center = self.real_target_center
        radius = self.real_target_radius
        height = self.real_target_height
        
        # Center points of top and bottom faces
        points.append(center + np.array([0, 0, height/2]))
        points.append(center + np.array([0, 0, -height/2]))
        
        # Circumference points (8 points each on top and bottom)
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        
        for angle in angles:
            x_offset = radius * np.cos(angle)
            y_offset = radius * np.sin(angle)
            points.append(center + np.array([x_offset, y_offset, height/2]))
            points.append(center + np.array([x_offset, y_offset, -height/2]))
            
        return points
    
    def _decode_decision_variables(self, x: np.ndarray) -> Dict[str, float]:
        """
        Decode optimization variables to named parameters
        
        Args:
            x: Decision variable vector [flight_angle, flight_speed, drop_time, detonation_delay]
            
        Returns:
            Dictionary of decoded parameters
        """
        return {
            'flight_angle': x[0],
            'flight_speed': x[1], 
            'drop_time': x[2],
            'detonation_delay': x[3]
        }
    
    def _calculate_drone_velocity_from_angle(self, angle: float, speed: float) -> np.ndarray:
        """
        Calculate 3D drone velocity vector from flight angle and speed
        
        Args:
            angle: Flight direction angle in xy-plane [radians]
            speed: Flight speed [m/s]
            
        Returns:
            3D velocity vector with zero z-component
        """
        return np.array([
            speed * np.cos(angle),
            speed * np.sin(angle),
            0.0  # Constant altitude flight
        ])
    
    def objective_function(self, x: np.ndarray) -> float:
        """
        Objective function to maximize obscuration duration
        
        Args:
            x: Decision variable vector
            
        Returns:
            Negative obscuration duration (for minimization)
        """
        self.evaluation_count += 1
        
        try:
            params = self._decode_decision_variables(x)
            
            # Calculate drone velocity vector
            drone_velocity = self._calculate_drone_velocity_from_angle(
                params['flight_angle'], params['flight_speed'])
            
            # Calculate obscuration duration
            duration = self.simulation_engine.calculate_obscuration_duration(
                missile_initial_pos=self.missile_initial_pos,
                missile_target_pos=self.fake_target_pos,
                drone_initial_pos=self.drone_initial_pos,
                drone_velocity=drone_velocity,
                drop_time=params['drop_time'],
                detonation_delay=params['detonation_delay'],
                target_points=self.target_key_points,
                max_simulation_time=120.0  # Reasonable upper bound
            )
            
            # Store good solutions for analysis
            if duration > 0:
                self.best_solutions.append({
                    'parameters': params.copy(),
                    'duration': duration,
                    'evaluation': self.evaluation_count
                })
            
            # Return negative for minimization
            return -duration
            
        except Exception as e:
            warnings.warn(f"Evaluation error: {e}")
            return 0.0  # Return worst possible value
    
    def solve_with_differential_evolution(self, max_iterations: int = 100, 
                                        population_size: int = 15) -> Dict[str, Any]:
        """
        Solve optimization using Differential Evolution algorithm
        
        Args:
            max_iterations: Maximum number of iterations
            population_size: Population size for DE algorithm
            
        Returns:
            Optimization results dictionary
        """
        print("使用差分进化算法求解问题2...")
        
        # Reset optimization tracking
        self.evaluation_count = 0
        self.best_solutions = []
        
        # Define bounds for scipy.optimize format
        bounds_list = [
            self.bounds['flight_angle'],
            self.bounds['flight_speed'], 
            self.bounds['drop_time'],
            self.bounds['detonation_delay']
        ]
        
        # Run differential evolution
        result = differential_evolution(
            self.objective_function,
            bounds_list,
            strategy='best1bin',
            maxiter=max_iterations,
            popsize=population_size,
            seed=42,  # For reproducibility
            disp=True,
            atol=1e-6,
            tol=1e-6
        )
        
        # Decode optimal solution
        optimal_params = self._decode_decision_variables(result.x)
        optimal_duration = -result.fun
        
        # Calculate additional analysis
        optimal_velocity = self._calculate_drone_velocity_from_angle(
            optimal_params['flight_angle'], optimal_params['flight_speed'])
        
        analysis = self._perform_solution_analysis(optimal_params, optimal_velocity)
        
        return {
            'optimization_result': result,
            'optimal_parameters': optimal_params,
            'optimal_duration': optimal_duration,
            'optimal_velocity': optimal_velocity,
            'analysis': analysis,
            'optimization_stats': {
                'total_evaluations': self.evaluation_count,
                'convergence_success': result.success,
                'convergence_message': result.message,
                'best_solutions_found': len(self.best_solutions)
            }
        }
    
    def solve_with_multi_start(self, n_starts: int = 5) -> Dict[str, Any]:
        """
        Solve optimization using multiple random starts for global search
        
        Args:
            n_starts: Number of random starting points
            
        Returns:
            Best optimization results from all starts
        """
        print(f"使用多起点优化算法求解问题2 (起点数: {n_starts})...")
        
        best_result = None
        best_duration = -np.inf
        all_results = []
        
        for start_idx in range(n_starts):
            print(f"  正在执行第 {start_idx + 1}/{n_starts} 次优化...")
            
            # Generate random starting point within bounds
            x0 = np.array([
                np.random.uniform(*self.bounds['flight_angle']),
                np.random.uniform(*self.bounds['flight_speed']),
                np.random.uniform(*self.bounds['drop_time']),
                np.random.uniform(*self.bounds['detonation_delay'])
            ])
            
            # Run local optimization from this starting point
            bounds_list = [
                self.bounds['flight_angle'],
                self.bounds['flight_speed'],
                self.bounds['drop_time'], 
                self.bounds['detonation_delay']
            ]
            
            result = minimize(
                self.objective_function,
                x0,
                method='L-BFGS-B',
                bounds=bounds_list,
                options={'maxiter': 100, 'disp': False}
            )
            
            current_duration = -result.fun
            all_results.append((result, current_duration))
            
            if current_duration > best_duration:
                best_duration = current_duration
                best_result = result
        
        # Process best result
        optimal_params = self._decode_decision_variables(best_result.x)
        optimal_velocity = self._calculate_drone_velocity_from_angle(
            optimal_params['flight_angle'], optimal_params['flight_speed'])
        analysis = self._perform_solution_analysis(optimal_params, optimal_velocity)
        
        return {
            'optimization_result': best_result,
            'optimal_parameters': optimal_params,
            'optimal_duration': best_duration,
            'optimal_velocity': optimal_velocity,
            'analysis': analysis,
            'optimization_stats': {
                'total_evaluations': self.evaluation_count,
                'n_starts': n_starts,
                'all_durations': [duration for _, duration in all_results],
                'convergence_success': best_result.success
            }
        }
    
    def _perform_solution_analysis(self, params: Dict[str, float], 
                                 velocity: np.ndarray) -> Dict[str, Any]:
        """
        Perform detailed analysis of optimal solution
        
        Args:
            params: Optimal parameters
            velocity: Optimal drone velocity vector
            
        Returns:
            Analysis results dictionary
        """
        detonation_time = params['drop_time'] + params['detonation_delay']
        
        # Calculate key positions
        drop_position = TrajectoryCalculator.drone_position(
            self.drone_initial_pos, velocity, params['drop_time'])
            
        detonation_position = TrajectoryCalculator.grenade_trajectory(
            drop_position, velocity, params['detonation_delay'])
        
        # Missile position at detonation
        missile_pos_at_detonation = TrajectoryCalculator.missile_position(
            self.missile_initial_pos, self.fake_target_pos,
            PhysicsConstants.MISSILE_SPEED, detonation_time)
        
        # Geometric analysis
        cloud_to_target_distance = Vector3D.distance(
            detonation_position, self.real_target_center)
        missile_to_target_at_detonation = Vector3D.distance(
            missile_pos_at_detonation, self.real_target_center)
        
        return {
            'timing': {
                'detonation_time': detonation_time,
                'cloud_lifetime_end': detonation_time + PhysicsConstants.CLOUD_LIFETIME
            },
            'positions': {
                'drop_position': drop_position,
                'detonation_position': detonation_position,
                'missile_at_detonation': missile_pos_at_detonation
            },
            'distances': {
                'cloud_to_target': cloud_to_target_distance,
                'missile_to_target_at_detonation': missile_to_target_at_detonation,
                'detonation_altitude': detonation_position[2]
            },
            'flight_direction_degrees': np.degrees(params['flight_angle']) % 360
        }
    
    def print_solution(self, results: Dict[str, Any]) -> None:
        """Print formatted optimization results"""
        print("=" * 70)
        print("问题2解决方案：单枚干扰弹的最优投放策略")
        print("=" * 70)
        
        print(f"\n最优结果：")
        print(f"最大有效遮蔽时长: {results['optimal_duration']:.3f} 秒")
        
        print(f"\n最优策略参数：")
        params = results['optimal_parameters']
        analysis = results['analysis']
        print(f"飞行方向角: {params['flight_angle']:.3f} 弧度 "
              f"({analysis['flight_direction_degrees']:.1f}°)")
        print(f"飞行速度: {params['flight_speed']:.1f} 米/秒")
        print(f"投放时间: {params['drop_time']:.2f} 秒")
        print(f"起爆延迟: {params['detonation_delay']:.2f} 秒")
        print(f"起爆时刻: {analysis['timing']['detonation_time']:.2f} 秒")
        
        print(f"\n几何分析：")
        print(f"起爆点位置: ({analysis['positions']['detonation_position'][0]:.1f}, "
              f"{analysis['positions']['detonation_position'][1]:.1f}, "
              f"{analysis['positions']['detonation_position'][2]:.1f}) 米")
        print(f"烟幕云团到真目标距离: {analysis['distances']['cloud_to_target']:.1f} 米")
        print(f"起爆时导弹到真目标距离: {analysis['distances']['missile_to_target_at_detonation']:.1f} 米")
        
        print(f"\n优化统计：")
        stats = results['optimization_stats']
        print(f"总评估次数: {stats['total_evaluations']}")
        print(f"收敛成功: {stats['convergence_success']}")

def main():
    """Main function to run Problem 2 solution"""
    print("正在求解问题2：单枚干扰弹的最优投放策略...")
    
    # Create solver
    solver = Problem2Solver(simulation_time_step=0.02)
    
    # Solve using differential evolution (recommended)
    results = solver.solve_with_differential_evolution(max_iterations=80, population_size=12)
    
    # Print results
    solver.print_solution(results)
    
    return results

if __name__ == "__main__":
    results = main()