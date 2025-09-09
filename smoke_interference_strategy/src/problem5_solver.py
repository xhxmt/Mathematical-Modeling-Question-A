"""
Problem 5: Multiple drones with multiple grenades against multiple missiles.

This module implements the most complex scenario: coordinating multiple drones,
each potentially deploying multiple grenades, to protect against multiple missiles.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.optimize import differential_evolution
import warnings
from itertools import product
from .physics_core import (
    PhysicsConstants, TrajectoryCalculator, ObscurationAnalyzer, 
    SimulationEngine, Vector3D
)

class MultiMissileSimulationEngine:
    """Advanced simulation engine for multiple missiles and multiple drone-grenade systems"""
    
    def __init__(self, time_step: float = 0.02):
        self.time_step = time_step
        
    def calculate_multi_missile_obscuration(self,
                                          missile_configs: List[Dict],
                                          drone_strategies: List[Dict],
                                          target_points: List[np.ndarray],
                                          max_simulation_time: float = 150.0) -> Dict[str, float]:
        """
        Calculate obscuration durations for multiple missiles
        
        Args:
            missile_configs: List of missile configurations
                Each dict contains: {
                    'missile_id': int,
                    'initial_position': np.ndarray,
                    'target_position': np.ndarray,
                    'speed': float
                }
            drone_strategies: List of drone strategies with grenade deployments
            target_points: Key points on target to protect
            max_simulation_time: Maximum simulation time
            
        Returns:
            Dictionary with obscuration durations for each missile and total
        """
        results = {}
        
        # Calculate cloud deployment schedule
        all_cloud_periods = []
        for strategy in drone_strategies:
            for grenade in strategy.get('grenades', []):
                drop_time = grenade['drop_time']
                detonation_delay = grenade['detonation_delay']
                detonation_time = drop_time + detonation_delay
                
                drop_position = TrajectoryCalculator.drone_position(
                    strategy['initial_position'], strategy['velocity'], drop_time)
                detonation_position = TrajectoryCalculator.grenade_trajectory(
                    drop_position, strategy['velocity'], detonation_delay)
                
                all_cloud_periods.append({
                    'drone_id': strategy['drone_id'],
                    'grenade_id': grenade.get('grenade_id', 1),
                    'detonation_time': detonation_time,
                    'detonation_position': detonation_position,
                    'fade_time': detonation_time + PhysicsConstants.CLOUD_LIFETIME
                })
        
        # Calculate obscuration for each missile
        total_obscuration = 0.0
        
        for missile_config in missile_configs:
            missile_id = missile_config['missile_id']
            missile_obscuration = self._calculate_single_missile_obscuration(
                missile_config, all_cloud_periods, target_points, max_simulation_time)
            
            results[f'missile_{missile_id}'] = missile_obscuration
            total_obscuration += missile_obscuration
        
        results['total_obscuration'] = total_obscuration
        results['average_obscuration'] = total_obscuration / len(missile_configs)
        
        return results
    
    def _calculate_single_missile_obscuration(self,
                                            missile_config: Dict,
                                            cloud_periods: List[Dict],
                                            target_points: List[np.ndarray],
                                            max_simulation_time: float) -> float:
        """Calculate obscuration duration for a single missile"""
        if not cloud_periods:
            return 0.0
            
        earliest_detonation = min(period['detonation_time'] for period in cloud_periods)
        latest_fade = max(period['fade_time'] for period in cloud_periods)
        simulation_end = min(max_simulation_time, latest_fade)
        
        obscuration_time = 0.0
        
        for t in np.arange(earliest_detonation, simulation_end, self.time_step):
            missile_pos = TrajectoryCalculator.missile_position(
                missile_config['initial_position'],
                missile_config['target_position'],
                missile_config['speed'], t)
            
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
                obscuration_time += self.time_step
                
        return obscuration_time

class Problem5Solver:
    """Solver for Problem 5: Multi-Drone Multi-Grenade vs Multi-Missile"""
    
    def __init__(self, num_drones: int = 3, grenades_per_drone: int = 3, 
                 num_missiles: int = 3, simulation_time_step: float = 0.03):
        """
        Initialize Problem 5 solver
        
        Args:
            num_drones: Number of drones
            grenades_per_drone: Number of grenades per drone
            num_missiles: Number of missiles to defend against
            simulation_time_step: Time step for simulation
        """
        self.num_drones = num_drones
        self.grenades_per_drone = grenades_per_drone
        self.num_missiles = num_missiles
        self.simulation_engine = MultiMissileSimulationEngine(simulation_time_step)
        
        # Multiple missile configurations (M1, M2, M3)
        self.missile_configs = [
            {
                'missile_id': 1,
                'initial_position': np.array([20000.0, 0.0, 2000.0]),
                'target_position': np.array([0.0, 0.0, 0.0]),
                'speed': PhysicsConstants.MISSILE_SPEED
            },
            {
                'missile_id': 2,
                'initial_position': np.array([19000.0, -1000.0, 2100.0]),
                'target_position': np.array([0.0, 0.0, 0.0]),
                'speed': PhysicsConstants.MISSILE_SPEED
            },
            {
                'missile_id': 3,
                'initial_position': np.array([19000.0, 1000.0, 1900.0]),
                'target_position': np.array([0.0, 0.0, 0.0]),
                'speed': PhysicsConstants.MISSILE_SPEED
            }
        ]
        
        # Multiple drone initial positions
        self.drone_initial_positions = [
            np.array([17800.0, 0.0, 1800.0]),    # FY1
            np.array([18000.0, -800.0, 1800.0]), # FY2
            np.array([18000.0, 800.0, 1800.0])   # FY3
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
        """Setup optimization bounds for all decision variables"""
        bounds = []
        
        # For each drone
        for i in range(self.num_drones):
            # Drone flight parameters
            bounds.extend([
                (0.0, 2*np.pi),    # flight_angle
                (70.0, 140.0),     # flight_speed
            ])
            
            # Grenade deployment parameters
            for j in range(self.grenades_per_drone):
                bounds.extend([
                    (0.1, 60.0),   # drop_time
                    (0.1, 8.0)     # detonation_delay
                ])
        
        return bounds
    
    def _decode_decision_variables(self, x: np.ndarray) -> Dict[str, Any]:
        """Decode optimization variables to drone strategies"""
        strategies = []
        var_index = 0
        
        for i in range(self.num_drones):
            # Decode drone flight parameters
            flight_angle = x[var_index]
            flight_speed = x[var_index + 1]
            var_index += 2
            
            # Create velocity vector
            velocity = np.array([
                flight_speed * np.cos(flight_angle),
                flight_speed * np.sin(flight_angle),
                0.0
            ])
            
            # Decode grenade parameters
            grenades = []
            for j in range(self.grenades_per_drone):
                drop_time = x[var_index]
                detonation_delay = x[var_index + 1]
                var_index += 2
                
                grenades.append({
                    'grenade_id': j + 1,
                    'drop_time': drop_time,
                    'detonation_delay': detonation_delay
                })
            
            # Sort and enforce minimum drop intervals
            grenades.sort(key=lambda g: g['drop_time'])
            for j in range(1, len(grenades)):
                min_time = grenades[j-1]['drop_time'] + 1.0
                if grenades[j]['drop_time'] < min_time:
                    grenades[j]['drop_time'] = min_time
            
            strategies.append({
                'drone_id': i + 1,
                'initial_position': self.drone_initial_positions[i],
                'flight_angle': flight_angle,
                'flight_speed': flight_speed,
                'velocity': velocity,
                'grenades': grenades
            })
        
        return {'drone_strategies': strategies}
    
    def objective_function(self, x: np.ndarray) -> float:
        """
        Multi-objective function: maximize total obscuration across all missiles
        """
        self.evaluation_count += 1
        
        try:
            params = self._decode_decision_variables(x)
            
            # Calculate obscuration results for all missiles
            obscuration_results = self.simulation_engine.calculate_multi_missile_obscuration(
                missile_configs=self.missile_configs,
                drone_strategies=params['drone_strategies'],
                target_points=self.target_key_points,
                max_simulation_time=120.0
            )
            
            # Use weighted total obscuration as objective
            # Priority weights: closer missiles get higher weight
            missile_weights = [1.0, 0.9, 0.9]  # M1 gets slightly higher priority
            
            weighted_total = sum(
                obscuration_results[f'missile_{i+1}'] * missile_weights[i]
                for i in range(self.num_missiles)
            )
            
            # Store promising solutions
            if weighted_total > 0:
                self.best_solutions.append({
                    'parameters': params.copy(),
                    'obscuration_results': obscuration_results.copy(),
                    'weighted_objective': weighted_total,
                    'evaluation': self.evaluation_count
                })
            
            return -weighted_total  # Negative for minimization
            
        except Exception as e:
            warnings.warn(f"Evaluation error: {e}")
            return 0.0
    
    def solve(self, max_iterations: int = 60, population_size: int = 25) -> Dict[str, Any]:
        """Solve Problem 5 using differential evolution with reduced complexity"""
        print(f"求解问题5：{self.num_drones}架无人机 × {self.grenades_per_drone}枚干扰弹 对 {self.num_missiles}枚导弹...")
        print(f"决策变量维度: {len(self.bounds)}")
        
        # Reset optimization tracking
        self.evaluation_count = 0
        self.best_solutions = []
        
        # Use more conservative optimization settings due to high dimensionality
        result = differential_evolution(
            self.objective_function,
            self.bounds,
            strategy='best1bin',
            maxiter=max_iterations,
            popsize=population_size,
            seed=42,
            disp=True,
            atol=1e-5,
            tol=1e-5
        )
        
        # Process results
        optimal_params = self._decode_decision_variables(result.x)
        
        # Re-evaluate optimal solution for detailed analysis
        optimal_obscuration = self.simulation_engine.calculate_multi_missile_obscuration(
            missile_configs=self.missile_configs,
            drone_strategies=optimal_params['drone_strategies'],
            target_points=self.target_key_points,
            max_simulation_time=120.0
        )
        
        # Calculate analysis
        analysis = self._perform_solution_analysis(optimal_params, optimal_obscuration)
        
        return {
            'optimization_result': result,
            'optimal_parameters': optimal_params,
            'optimal_obscuration': optimal_obscuration,
            'analysis': analysis,
            'optimization_stats': {
                'total_evaluations': self.evaluation_count,
                'convergence_success': result.success,
                'best_solutions_found': len(self.best_solutions)
            }
        }
    
    def _perform_solution_analysis(self, params: Dict[str, Any], 
                                 obscuration_results: Dict[str, float]) -> Dict[str, Any]:
        """Perform comprehensive analysis of optimal solution"""
        
        # Analyze drone deployments
        drone_analysis = []
        all_deployments = []
        
        for strategy in params['drone_strategies']:
            drone_info = {
                'drone_id': strategy['drone_id'],
                'initial_position': strategy['initial_position'],
                'flight_angle_degrees': np.degrees(strategy['flight_angle']) % 360,
                'flight_speed': strategy['flight_speed'],
                'grenades': []
            }
            
            for grenade in strategy['grenades']:
                drop_time = grenade['drop_time']
                detonation_delay = grenade['detonation_delay']
                detonation_time = drop_time + detonation_delay
                
                drop_position = TrajectoryCalculator.drone_position(
                    strategy['initial_position'], strategy['velocity'], drop_time)
                detonation_position = TrajectoryCalculator.grenade_trajectory(
                    drop_position, strategy['velocity'], detonation_delay)
                
                grenade_info = {
                    'grenade_id': grenade['grenade_id'],
                    'drop_time': drop_time,
                    'detonation_delay': detonation_delay,
                    'detonation_time': detonation_time,
                    'detonation_position': detonation_position,
                    'cloud_fade_time': detonation_time + PhysicsConstants.CLOUD_LIFETIME
                }
                
                drone_info['grenades'].append(grenade_info)
                all_deployments.append({
                    'drone_id': strategy['drone_id'],
                    'grenade_id': grenade['grenade_id'],
                    'detonation_time': detonation_time,
                    'fade_time': grenade_info['cloud_fade_time']
                })
            
            drone_analysis.append(drone_info)
        
        # Global timing analysis
        if all_deployments:
            timing_analysis = {
                'earliest_detonation': min(d['detonation_time'] for d in all_deployments),
                'latest_detonation': max(d['detonation_time'] for d in all_deployments),
                'earliest_fade': min(d['fade_time'] for d in all_deployments),
                'latest_fade': max(d['fade_time'] for d in all_deployments),
                'total_coverage_span': max(d['fade_time'] for d in all_deployments) - 
                                     min(d['detonation_time'] for d in all_deployments)
            }
        else:
            timing_analysis = {}
        
        return {
            'drone_deployments': drone_analysis,
            'missile_obscuration_results': obscuration_results,
            'timing_analysis': timing_analysis,
            'performance_metrics': {
                'total_grenades_deployed': sum(len(d['grenades']) for d in drone_analysis),
                'average_obscuration_per_missile': obscuration_results.get('average_obscuration', 0),
                'best_protected_missile': max(
                    [(i+1, obscuration_results[f'missile_{i+1}']) 
                     for i in range(self.num_missiles)],
                    key=lambda x: x[1]) if obscuration_results else (None, 0)
            }
        }
    
    def print_solution(self, results: Dict[str, Any]) -> None:
        """Print comprehensive solution results"""
        print("=" * 90)
        print(f"问题5解决方案：{self.num_drones}架无人机×{self.grenades_per_drone}枚干扰弹 对 {self.num_missiles}枚导弹")
        print("=" * 90)
        
        obscuration = results['optimal_obscuration']
        
        print(f"\n导弹防护效果：")
        for i in range(self.num_missiles):
            duration = obscuration[f'missile_{i+1}']
            print(f"导弹 M{i+1}: {duration:.3f} 秒")
        print(f"总遮蔽时长: {obscuration['total_obscuration']:.3f} 秒")
        print(f"平均遮蔽时长: {obscuration['average_obscuration']:.3f} 秒")
        
        print(f"\n各无人机部署详情：")
        for drone in results['analysis']['drone_deployments']:
            print(f"无人机 FY{drone['drone_id']}:")
            print(f"  飞行方向: {drone['flight_angle_degrees']:.1f}°")
            print(f"  飞行速度: {drone['flight_speed']:.1f} 米/秒")
            print(f"  干扰弹部署：")
            for grenade in drone['grenades']:
                print(f"    干扰弹 {grenade['grenade_id']}: "
                      f"投放 {grenade['drop_time']:.1f}s, "
                      f"起爆 {grenade['detonation_time']:.1f}s")
        
        if results['analysis']['timing_analysis']:
            timing = results['analysis']['timing_analysis']
            print(f"\n协调时序分析：")
            print(f"首次起爆: {timing['earliest_detonation']:.1f} 秒")
            print(f"最后起爆: {timing['latest_detonation']:.1f} 秒")
            print(f"最后消散: {timing['latest_fade']:.1f} 秒")
            print(f"总覆盖时长: {timing['total_coverage_span']:.1f} 秒")
        
        metrics = results['analysis']['performance_metrics']
        best_missile, best_duration = metrics['best_protected_missile']
        print(f"\n性能指标：")
        print(f"部署干扰弹总数: {metrics['total_grenades_deployed']}")
        print(f"最佳防护效果: 导弹 M{best_missile} ({best_duration:.3f}秒)")

def main():
    """Main function to run Problem 5 solution"""
    print("正在求解问题5：多无人机多干扰弹对多导弹的综合防护策略...")
    
    # Create solver with reduced complexity for demonstration
    solver = Problem5Solver(num_drones=3, grenades_per_drone=2, num_missiles=3, 
                           simulation_time_step=0.03)
    
    # Solve with conservative settings
    results = solver.solve(max_iterations=50, population_size=20)
    
    # Print results
    solver.print_solution(results)
    
    return results

if __name__ == "__main__":
    results = main()