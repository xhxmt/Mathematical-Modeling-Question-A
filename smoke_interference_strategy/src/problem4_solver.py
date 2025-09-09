"""
Problem 4: Multiple drones with single grenade each deployment strategy.

This module implements optimization for multiple drones, each deploying one grenade,
to maximize total obscuration duration against a single missile.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.optimize import differential_evolution
import warnings
from .physics_core import (
    PhysicsConstants, TrajectoryCalculator, ObscurationAnalyzer, 
    SimulationEngine, Vector3D
)

class MultiDroneSimulationEngine:
    """Simulation engine for multiple drones with smoke grenades"""
    
    def __init__(self, time_step: float = 0.02):
        self.time_step = time_step
        
    def calculate_multi_drone_obscuration_duration(self,
                                                 missile_initial_pos: np.ndarray,
                                                 missile_target_pos: np.ndarray,
                                                 drone_strategies: List[Dict],
                                                 target_points: List[np.ndarray],
                                                 max_simulation_time: float = 150.0) -> float:
        """
        Calculate total obscuration duration for multiple drone-grenade pairs
        
        Args:
            missile_initial_pos: Initial missile position
            missile_target_pos: Missile target position
            drone_strategies: List of drone deployment strategies
                Each dict contains: {
                    'initial_position': np.ndarray,
                    'velocity': np.ndarray, 
                    'drop_time': float,
                    'detonation_delay': float
                }
            target_points: Key points on target to protect
            max_simulation_time: Maximum simulation time
            
        Returns:
            Total obscuration duration considering all cloud overlaps
        """
        # Calculate cloud parameters for each drone-grenade pair
        cloud_periods = []
        
        for i, strategy in enumerate(drone_strategies):
            drop_time = strategy['drop_time']
            detonation_delay = strategy['detonation_delay']
            detonation_time = drop_time + detonation_delay
            
            # Calculate positions for this drone
            drop_position = TrajectoryCalculator.drone_position(
                strategy['initial_position'], strategy['velocity'], drop_time)
            detonation_position = TrajectoryCalculator.grenade_trajectory(
                drop_position, strategy['velocity'], detonation_delay)
            
            cloud_periods.append({
                'drone_id': i + 1,
                'detonation_time': detonation_time,
                'detonation_position': detonation_position,
                'fade_time': detonation_time + PhysicsConstants.CLOUD_LIFETIME
            })
        
        # Find simulation window
        if not cloud_periods:
            return 0.0
            
        earliest_detonation = min(period['detonation_time'] for period in cloud_periods)
        latest_fade = max(period['fade_time'] for period in cloud_periods)
        simulation_end = min(max_simulation_time, latest_fade)
        
        # Time-step simulation
        total_obscuration_time = 0.0
        
        for t in np.arange(earliest_detonation, simulation_end, self.time_step):
            missile_pos = TrajectoryCalculator.missile_position(
                missile_initial_pos, missile_target_pos,
                PhysicsConstants.MISSILE_SPEED, t)
            
            # Check if any active cloud provides obscuration
            is_obscured = False
            
            for period in cloud_periods:
                if period['detonation_time'] <= t <= period['fade_time']:
                    time_since_detonation = t - period['detonation_time']
                    cloud_center = TrajectoryCalculator.smoke_cloud_center(
                        period['detonation_position'], time_since_detonation)
                    
                    if ObscurationAnalyzer.is_target_obscured(
                        missile_pos, target_points, cloud_center, 
                        PhysicsConstants.CLOUD_RADIUS):
                        is_obscured = True
                        break
            
            if is_obscured:
                total_obscuration_time += self.time_step
                
        return total_obscuration_time

class Problem4Solver:
    """Solver for Problem 4: Multiple Drones Single Grenade Each Strategy"""
    
    def __init__(self, num_drones: int = 3, simulation_time_step: float = 0.02):
        """
        Initialize Problem 4 solver
        
        Args:
            num_drones: Number of drones to coordinate
            simulation_time_step: Time step for numerical simulation
        """
        self.num_drones = num_drones
        self.simulation_engine = MultiDroneSimulationEngine(simulation_time_step)
        
        # Fixed parameters
        self.missile_initial_pos = np.array([20000.0, 0.0, 2000.0])
        self.fake_target_pos = np.array([0.0, 0.0, 0.0])
        
        # Initial positions for multiple drones (FY1, FY2, FY3)
        self.drone_initial_positions = [
            np.array([17800.0, 0.0, 1800.0]),    # FY1
            np.array([18000.0, -500.0, 1800.0]), # FY2 (offset position)
            np.array([18000.0, 500.0, 1800.0])   # FY3 (offset position)
        ]
        
        # Target definition
        self.real_target_center = np.array([0.0, 200.0, 5.0])
        self.real_target_radius = 200.0
        self.real_target_height = 10.0
        self.target_key_points = self._generate_target_key_points()
        
        # Decision variable bounds
        self.bounds = self._setup_optimization_bounds()
        
        # Optimization tracking
        self.evaluation_count = 0
        self.best_solutions = []
    
    def _generate_target_key_points(self) -> List[np.ndarray]:
        """Generate key points on the cylindrical real target"""
        points = []
        center = self.real_target_center
        radius = self.real_target_radius
        height = self.real_target_height
        
        points.append(center + np.array([0, 0, height/2]))
        points.append(center + np.array([0, 0, -height/2]))
        
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)
        for angle in angles:
            x_offset = radius * np.cos(angle)
            y_offset = radius * np.sin(angle)
            points.append(center + np.array([x_offset, y_offset, height/2]))
            points.append(center + np.array([x_offset, y_offset, -height/2]))
            
        return points
    
    def _setup_optimization_bounds(self) -> List[Tuple[float, float]]:
        """
        Setup optimization bounds for decision variables
        
        For each drone: [flight_angle, flight_speed, drop_time, detonation_delay]
        """
        bounds = []
        
        for i in range(self.num_drones):
            bounds.extend([
                (0.0, 2*np.pi),    # flight_angle_i
                (70.0, 140.0),     # flight_speed_i
                (0.1, 60.0),       # drop_time_i
                (0.1, 10.0)        # detonation_delay_i
            ])
            
        return bounds
    
    def _decode_decision_variables(self, x: np.ndarray) -> Dict[str, Any]:
        """Decode optimization variables to drone strategies"""
        strategies = []
        
        for i in range(self.num_drones):
            base_idx = i * 4
            strategy = {
                'drone_id': i + 1,
                'initial_position': self.drone_initial_positions[i],
                'flight_angle': x[base_idx],
                'flight_speed': x[base_idx + 1],
                'drop_time': x[base_idx + 2],
                'detonation_delay': x[base_idx + 3]
            }
            
            # Calculate velocity vector
            strategy['velocity'] = np.array([
                strategy['flight_speed'] * np.cos(strategy['flight_angle']),
                strategy['flight_speed'] * np.sin(strategy['flight_angle']),
                0.0
            ])
            
            strategies.append(strategy)
        
        return {'drone_strategies': strategies}
    
    def objective_function(self, x: np.ndarray) -> float:
        """Objective function to maximize total obscuration duration"""
        self.evaluation_count += 1
        
        try:
            params = self._decode_decision_variables(x)
            
            # Calculate total obscuration duration
            duration = self.simulation_engine.calculate_multi_drone_obscuration_duration(
                missile_initial_pos=self.missile_initial_pos,
                missile_target_pos=self.fake_target_pos,
                drone_strategies=params['drone_strategies'],
                target_points=self.target_key_points,
                max_simulation_time=150.0
            )
            
            # Store promising solutions
            if duration > 0:
                self.best_solutions.append({
                    'parameters': params.copy(),
                    'duration': duration,
                    'evaluation': self.evaluation_count
                })
            
            return -duration  # Negative for minimization
            
        except Exception as e:
            warnings.warn(f"Evaluation error: {e}")
            return 0.0
    
    def solve(self, max_iterations: int = 100, population_size: int = 25) -> Dict[str, Any]:
        """Solve Problem 4 using differential evolution"""
        print(f"使用差分进化算法求解问题4 (协调 {self.num_drones} 架无人机)...")
        
        # Reset optimization tracking
        self.evaluation_count = 0
        self.best_solutions = []
        
        # Run differential evolution
        result = differential_evolution(
            self.objective_function,
            self.bounds,
            strategy='best1bin',
            maxiter=max_iterations,
            popsize=population_size,
            seed=42,
            disp=True,
            atol=1e-6,
            tol=1e-6
        )
        
        # Process results
        optimal_params = self._decode_decision_variables(result.x)
        optimal_duration = -result.fun
        
        # Calculate analysis
        analysis = self._perform_solution_analysis(optimal_params)
        
        return {
            'optimization_result': result,
            'optimal_parameters': optimal_params,
            'optimal_duration': optimal_duration,
            'analysis': analysis,
            'optimization_stats': {
                'total_evaluations': self.evaluation_count,
                'convergence_success': result.success,
                'best_solutions_found': len(self.best_solutions)
            }
        }
    
    def _perform_solution_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis of optimal solution"""
        
        drone_analysis = []
        
        for strategy in params['drone_strategies']:
            drop_time = strategy['drop_time']
            detonation_delay = strategy['detonation_delay']
            detonation_time = drop_time + detonation_delay
            
            # Calculate positions
            drop_position = TrajectoryCalculator.drone_position(
                strategy['initial_position'], strategy['velocity'], drop_time)
            detonation_position = TrajectoryCalculator.grenade_trajectory(
                drop_position, strategy['velocity'], detonation_delay)
            
            drone_analysis.append({
                'drone_id': strategy['drone_id'],
                'initial_position': strategy['initial_position'],
                'flight_angle_degrees': np.degrees(strategy['flight_angle']) % 360,
                'flight_speed': strategy['flight_speed'],
                'drop_time': drop_time,
                'detonation_delay': detonation_delay,
                'detonation_time': detonation_time,
                'drop_position': drop_position,
                'detonation_position': detonation_position,
                'cloud_fade_time': detonation_time + PhysicsConstants.CLOUD_LIFETIME,
                'cloud_to_target_distance': Vector3D.distance(
                    detonation_position, self.real_target_center)
            })
        
        return {
            'drone_deployments': drone_analysis,
            'coordination_analysis': {
                'earliest_drop': min(d['drop_time'] for d in drone_analysis),
                'latest_drop': max(d['drop_time'] for d in drone_analysis),
                'earliest_detonation': min(d['detonation_time'] for d in drone_analysis),
                'latest_detonation': max(d['detonation_time'] for d in drone_analysis),
                'coverage_timespan': max(d['cloud_fade_time'] for d in drone_analysis) - 
                                   min(d['detonation_time'] for d in drone_analysis)
            }
        }
    
    def print_solution(self, results: Dict[str, Any]) -> None:
        """Print formatted optimization results"""
        print("=" * 80)
        print(f"问题4解决方案：{self.num_drones} 架无人机协调部署策略")
        print("=" * 80)
        
        print(f"\n最优结果：")
        print(f"最大总遮蔽时长: {results['optimal_duration']:.3f} 秒")
        
        print(f"\n各无人机部署策略：")
        for drone in results['analysis']['drone_deployments']:
            print(f"无人机 FY{drone['drone_id']}:")
            print(f"  初始位置: ({drone['initial_position'][0]:.0f}, "
                  f"{drone['initial_position'][1]:.0f}, "
                  f"{drone['initial_position'][2]:.0f}) 米")
            print(f"  飞行方向: {drone['flight_angle_degrees']:.1f}°")
            print(f"  飞行速度: {drone['flight_speed']:.1f} 米/秒")
            print(f"  投放时间: {drone['drop_time']:.2f} 秒")
            print(f"  起爆延迟: {drone['detonation_delay']:.2f} 秒")
            print(f"  起爆时刻: {drone['detonation_time']:.2f} 秒")
            print(f"  到真目标距离: {drone['cloud_to_target_distance']:.1f} 米")
        
        print(f"\n协调分析：")
        coord = results['analysis']['coordination_analysis']
        print(f"最早投放: {coord['earliest_drop']:.2f} 秒")
        print(f"最晚投放: {coord['latest_drop']:.2f} 秒")
        print(f"最早起爆: {coord['earliest_detonation']:.2f} 秒")
        print(f"最晚起爆: {coord['latest_detonation']:.2f} 秒")
        print(f"总覆盖时长: {coord['coverage_timespan']:.2f} 秒")

def main():
    """Main function to run Problem 4 solution"""
    print("正在求解问题4：多无人机单干扰弹协调部署策略...")
    
    # Create solver with 3 drones
    solver = Problem4Solver(num_drones=3, simulation_time_step=0.02)
    
    # Solve optimization problem
    results = solver.solve(max_iterations=80, population_size=20)
    
    # Print results
    solver.print_solution(results)
    
    return results

if __name__ == "__main__":
    results = main()