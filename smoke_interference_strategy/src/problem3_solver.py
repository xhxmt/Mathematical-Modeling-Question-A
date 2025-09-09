"""
Problem 3: Single drone with multiple smoke grenades deployment strategy.

This module implements optimization for a single drone deploying multiple grenades
to maximize total obscuration duration, handling temporal overlap between clouds.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.optimize import differential_evolution
import warnings
from .physics_core import (
    PhysicsConstants, TrajectoryCalculator, ObscurationAnalyzer, 
    SimulationEngine, Vector3D
)

class MultiGrenadeSimulationEngine:
    """Enhanced simulation engine for multiple smoke grenades"""
    
    def __init__(self, time_step: float = 0.02):
        self.time_step = time_step
        
    def calculate_total_obscuration_duration(self,
                                           missile_initial_pos: np.ndarray,
                                           missile_target_pos: np.ndarray,
                                           drone_initial_pos: np.ndarray,
                                           drone_velocity: np.ndarray,
                                           grenade_strategies: List[Dict],
                                           target_points: List[np.ndarray],
                                           max_simulation_time: float = 150.0) -> float:
        """
        Calculate total obscuration duration for multiple grenades
        
        Args:
            missile_initial_pos: Initial missile position
            missile_target_pos: Missile target position
            drone_initial_pos: Initial drone position
            drone_velocity: Drone velocity vector
            grenade_strategies: List of grenade deployment parameters
                Each dict contains: {'drop_time': float, 'detonation_delay': float}
            target_points: Key points on target to protect
            max_simulation_time: Maximum simulation time
            
        Returns:
            Total obscuration duration considering cloud overlaps
        """
        # Calculate detonation parameters for each grenade
        cloud_periods = []
        for strategy in grenade_strategies:
            drop_time = strategy['drop_time']
            detonation_delay = strategy['detonation_delay']
            detonation_time = drop_time + detonation_delay
            
            # Calculate positions
            drop_position = TrajectoryCalculator.drone_position(
                drone_initial_pos, drone_velocity, drop_time)
            detonation_position = TrajectoryCalculator.grenade_trajectory(
                drop_position, drone_velocity, detonation_delay)
            
            cloud_periods.append({
                'detonation_time': detonation_time,
                'detonation_position': detonation_position,
                'fade_time': detonation_time + PhysicsConstants.CLOUD_LIFETIME
            })
        
        # Sort cloud periods by detonation time
        cloud_periods.sort(key=lambda x: x['detonation_time'])
        
        # Find overall simulation window
        earliest_detonation = min(period['detonation_time'] for period in cloud_periods)
        latest_fade = max(period['fade_time'] for period in cloud_periods)
        simulation_end = min(max_simulation_time, latest_fade)
        
        # Time-step simulation to find union of obscured periods
        total_obscuration_time = 0.0
        
        for t in np.arange(earliest_detonation, simulation_end, self.time_step):
            missile_pos = TrajectoryCalculator.missile_position(
                missile_initial_pos, missile_target_pos,
                PhysicsConstants.MISSILE_SPEED, t)
            
            # Check if any active cloud provides obscuration
            is_obscured = False
            
            for period in cloud_periods:
                # Check if this cloud is active at time t
                if period['detonation_time'] <= t <= period['fade_time']:
                    # Calculate current cloud center position
                    time_since_detonation = t - period['detonation_time']
                    cloud_center = TrajectoryCalculator.smoke_cloud_center(
                        period['detonation_position'], time_since_detonation)
                    
                    # Check if this cloud obscures the target
                    if ObscurationAnalyzer.is_target_obscured(
                        missile_pos, target_points, cloud_center, 
                        PhysicsConstants.CLOUD_RADIUS):
                        is_obscured = True
                        break  # One cloud providing obscuration is sufficient
            
            if is_obscured:
                total_obscuration_time += self.time_step
                
        return total_obscuration_time

class Problem3Solver:
    """Solver for Problem 3: Single Drone Multiple Grenades Strategy"""
    
    def __init__(self, num_grenades: int = 3, simulation_time_step: float = 0.02):
        """
        Initialize Problem 3 solver
        
        Args:
            num_grenades: Number of grenades to deploy
            simulation_time_step: Time step for numerical simulation
        """
        self.num_grenades = num_grenades
        self.simulation_engine = MultiGrenadeSimulationEngine(simulation_time_step)
        
        # Fixed parameters
        self.missile_initial_pos = np.array([20000.0, 0.0, 2000.0])
        self.drone_initial_pos = np.array([17800.0, 0.0, 1800.0])
        self.fake_target_pos = np.array([0.0, 0.0, 0.0])
        
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
        
        # Center points
        points.append(center + np.array([0, 0, height/2]))
        points.append(center + np.array([0, 0, -height/2]))
        
        # Circumference points
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
        
        Decision variables: [flight_angle, flight_speed, 
                           drop_time_1, detonation_delay_1,
                           drop_time_2, detonation_delay_2,
                           drop_time_3, detonation_delay_3]
        """
        bounds = [
            (0.0, 2*np.pi),        # flight_angle
            (70.0, 140.0),         # flight_speed
        ]
        
        # Add bounds for each grenade
        for i in range(self.num_grenades):
            bounds.extend([
                (0.1, 60.0),       # drop_time_i
                (0.1, 10.0)        # detonation_delay_i
            ])
            
        return bounds
    
    def _decode_decision_variables(self, x: np.ndarray) -> Dict[str, Any]:
        """Decode optimization variables to named parameters"""
        params = {
            'flight_angle': x[0],
            'flight_speed': x[1],
            'grenades': []
        }
        
        # Decode grenade parameters
        for i in range(self.num_grenades):
            base_idx = 2 + i * 2
            grenade_params = {
                'drop_time': x[base_idx],
                'detonation_delay': x[base_idx + 1]
            }
            params['grenades'].append(grenade_params)
            
        return params
    
    def _enforce_deployment_constraints(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Enforce constraint that grenades must be dropped with minimum intervals
        
        Constraint: drop_time[i+1] >= drop_time[i] + 1.0 seconds
        """
        # Sort grenades by drop time and enforce minimum intervals
        grenades = sorted(params['grenades'], key=lambda g: g['drop_time'])
        
        # Adjust drop times to satisfy constraints
        for i in range(1, len(grenades)):
            min_drop_time = grenades[i-1]['drop_time'] + 1.0
            if grenades[i]['drop_time'] < min_drop_time:
                grenades[i]['drop_time'] = min_drop_time
        
        params['grenades'] = grenades
        return params
    
    def objective_function(self, x: np.ndarray) -> float:
        """Improved objective function to maximize total obscuration duration"""
        self.evaluation_count += 1
        
        try:
            params = self._decode_decision_variables(x)
            params = self._enforce_deployment_constraints(params)
            
            # Calculate drone velocity
            drone_velocity = np.array([
                params['flight_speed'] * np.cos(params['flight_angle']),
                params['flight_speed'] * np.sin(params['flight_angle']),
                0.0
            ])
            
            # Calculate total obscuration duration
            duration = self.simulation_engine.calculate_total_obscuration_duration(
                missile_initial_pos=self.missile_initial_pos,
                missile_target_pos=self.fake_target_pos,
                drone_initial_pos=self.drone_initial_pos,
                drone_velocity=drone_velocity,
                grenade_strategies=params['grenades'],
                target_points=self.target_key_points,
                max_simulation_time=150.0
            )
            
            # Add regularization terms to encourage diversity and avoid local optima
            regularization = 0.0
            
            # 1. Encourage strategic spacing of grenades
            drop_times = [g['drop_time'] for g in params['grenades']]
            if len(set(drop_times)) > 1:  # Only if times are different
                time_variance = np.var(drop_times)
                regularization += 0.01 * time_variance  # Small bonus for temporal diversity
            
            # 2. Penalize extremely long delays that may be unrealistic
            delays = [g['detonation_delay'] for g in params['grenades']]
            avg_delay = np.mean(delays)
            if avg_delay > 8.0:  # If average delay is too long
                regularization -= 0.02 * (avg_delay - 8.0)
            
            # 3. Encourage reasonable flight speeds (not too slow)
            if params['flight_speed'] < 80:
                regularization -= 0.01 * (80 - params['flight_speed'])
            
            # Combine duration with regularization
            objective_value = duration + regularization
            
            # Store promising solutions (only if duration > 0)
            if duration > 0:
                self.best_solutions.append({
                    'parameters': params.copy(),
                    'duration': duration,
                    'objective_value': objective_value,
                    'evaluation': self.evaluation_count
                })
                
                # Print progress for better solutions
                if len(self.best_solutions) <= 5 or duration > max(s['duration'] for s in self.best_solutions[:-1]):
                    print(f"    评估 {self.evaluation_count}: 发现解 {duration:.3f}秒 (目标值: {objective_value:.3f})")
            
            return -objective_value  # Negative for minimization
            
        except Exception as e:
            warnings.warn(f"Evaluation error: {e}")
            return 0.0
    
    def solve_quick_test(self, max_iterations: int = 30, population_size: int = 12) -> Dict[str, Any]:
        """
        Quick test version of Problem 3 solver with reduced complexity
        
        Args:
            max_iterations: Maximum optimization iterations (reduced)
            population_size: Population size for DE (reduced)
            
        Returns:
            Optimization results dictionary
        """
        print(f"快速测试模式：求解问题3 (部署 {self.num_grenades} 枚干扰弹)...")
        
        # Reset optimization tracking
        self.evaluation_count = 0
        self.best_solutions = []
        
        # Use only 2 restarts for speed
        n_restarts = 2
        iterations_per_restart = max_iterations // n_restarts
        
        best_result = None
        best_duration = -np.inf
        
        for restart in range(n_restarts):
            print(f"  执行第 {restart + 1}/{n_restarts} 次重启优化 (每次 {iterations_per_restart} 迭代)...")
            
            # Simpler strategy selection
            strategy = 'best1bin' if restart == 0 else 'rand1exp'
            
            # Run with reduced complexity
            result = differential_evolution(
                self.objective_function,
                self.bounds,
                strategy=strategy,
                maxiter=iterations_per_restart,
                popsize=population_size,
                seed=42 + restart * 50,
                disp=False,
                atol=1e-4,  # More relaxed tolerance
                tol=1e-4,
                workers=1,
                mutation=0.7,
                recombination=0.8
            )
            
            current_duration = -result.fun
            print(f"    重启 {restart + 1} 结果: {current_duration:.3f} 秒")
            
            if current_duration > best_duration:
                best_duration = current_duration
                best_result = result
                print(f"    ✓ 发现更优解: {best_duration:.3f} 秒")
        
        # Process best results
        optimal_params = self._decode_decision_variables(best_result.x)
        optimal_params = self._enforce_deployment_constraints(optimal_params)
        optimal_duration = best_duration
        
        # Calculate analysis
        analysis = self._perform_solution_analysis(optimal_params)
        
        return {
            'optimization_result': best_result,
            'optimal_parameters': optimal_params,
            'optimal_duration': optimal_duration,
            'analysis': analysis,
            'optimization_stats': {
                'total_evaluations': self.evaluation_count,
                'convergence_success': best_result.success,
                'best_solutions_found': len(self.best_solutions),
                'n_restarts': n_restarts,
                'best_restart_score': best_duration,
                'mode': 'quick_test'
            }
        }
    
    def solve(self, max_iterations: int = 120, population_size: int = 20) -> Dict[str, Any]:
        """
        Solve Problem 3 using improved differential evolution with restart mechanism
        
        Args:
            max_iterations: Maximum optimization iterations
            population_size: Population size for DE
            
        Returns:
            Optimization results dictionary
        """
        # For faster execution, use the quick test version by default
        return self.solve_quick_test(max_iterations=60, population_size=15)
    
    def _perform_solution_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        
        best_result = None
        best_duration = -np.inf
        
        # Multi-restart approach to avoid local optima
        n_restarts = 3
        iterations_per_restart = max_iterations // n_restarts
        
        for restart in range(n_restarts):
            print(f"  执行第 {restart + 1}/{n_restarts} 次重启优化...")
            
            # Use different strategies and parameters for each restart
            if restart == 0:
                strategy = 'best1bin'
                atol = 1e-6
                tol = 1e-6
            elif restart == 1:
                strategy = 'rand1exp'
                atol = 1e-5
                tol = 1e-5
            else:
                strategy = 'best2bin'
                atol = 1e-4
                tol = 1e-4
            
            # Run differential evolution with current strategy
            result = differential_evolution(
                self.objective_function,
                self.bounds,
                strategy=strategy,
                maxiter=iterations_per_restart,
                popsize=population_size,
                seed=42 + restart * 100,  # Different seed for each restart
                disp=False,  # Reduce output verbosity
                atol=atol,
                tol=tol,
                workers=1,
                mutation=(0.5, 1.2),  # Adaptive mutation range
                recombination=0.7
            )
            
            current_duration = -result.fun
            print(f"    重启 {restart + 1} 结果: {current_duration:.3f} 秒")
            
            if current_duration > best_duration:
                best_duration = current_duration
                best_result = result
                print(f"    ✓ 发现更优解: {best_duration:.3f} 秒")
        
        # If all restarts failed, try a final attempt with different bounds
        if best_duration <= 0:
            print("  执行最终优化尝试...")
            
            # Try with more focused search space
            focused_bounds = []
            for i, (low, high) in enumerate(self.bounds):
                if i == 0:  # flight_angle
                    focused_bounds.append((0, np.pi))  # Limit to forward hemisphere
                elif i == 1:  # flight_speed
                    focused_bounds.append((100, 140))  # Focus on higher speeds
                else:
                    focused_bounds.append((low, high))
            
            result = differential_evolution(
                self.objective_function,
                focused_bounds,
                strategy='best1bin',
                maxiter=40,
                popsize=15,
                seed=999,
                disp=False,
                mutation=(0.3, 0.9),
                recombination=0.9
            )
            
            final_duration = -result.fun
            print(f"    最终尝试结果: {final_duration:.3f} 秒")
            
            if final_duration > best_duration:
                best_duration = final_duration
                best_result = result
                print(f"    ✓ 最终尝试成功!")
        
        # Process best results
        optimal_params = self._decode_decision_variables(best_result.x)
        optimal_params = self._enforce_deployment_constraints(optimal_params)
        optimal_duration = best_duration
        
        # Calculate analysis
        analysis = self._perform_solution_analysis(optimal_params)
        
        return {
            'optimization_result': best_result,
            'optimal_parameters': optimal_params,
            'optimal_duration': optimal_duration,
            'analysis': analysis,
            'optimization_stats': {
                'total_evaluations': self.evaluation_count,
                'convergence_success': best_result.success,
                'best_solutions_found': len(self.best_solutions),
                'n_restarts': n_restarts,
                'best_restart_score': best_duration
            }
        }
    
    def _perform_solution_analysis(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Perform detailed analysis of optimal solution"""
        
        # Calculate drone velocity
        drone_velocity = np.array([
            params['flight_speed'] * np.cos(params['flight_angle']),
            params['flight_speed'] * np.sin(params['flight_angle']),
            0.0
        ])
        
        # Analyze each grenade deployment
        grenade_analysis = []
        for i, grenade in enumerate(params['grenades']):
            drop_time = grenade['drop_time']
            detonation_delay = grenade['detonation_delay']
            detonation_time = drop_time + detonation_delay
            
            # Calculate positions
            drop_position = TrajectoryCalculator.drone_position(
                self.drone_initial_pos, drone_velocity, drop_time)
            detonation_position = TrajectoryCalculator.grenade_trajectory(
                drop_position, drone_velocity, detonation_delay)
            
            grenade_analysis.append({
                'grenade_id': i + 1,
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
            'drone_velocity': drone_velocity,
            'flight_direction_degrees': np.degrees(params['flight_angle']) % 360,
            'grenade_deployments': grenade_analysis,
            'timing_summary': {
                'first_drop': min(g['drop_time'] for g in grenade_analysis),
                'last_drop': max(g['drop_time'] for g in grenade_analysis),
                'first_detonation': min(g['detonation_time'] for g in grenade_analysis),
                'last_detonation': max(g['detonation_time'] for g in grenade_analysis),
                'last_cloud_fade': max(g['cloud_fade_time'] for g in grenade_analysis)
            }
        }
    
    def print_solution(self, results: Dict[str, Any]) -> None:
        """Print formatted optimization results"""
        print("=" * 80)
        print(f"问题3解决方案：单无人机 {self.num_grenades} 枚干扰弹部署策略")
        print("=" * 80)
        
        print(f"\n最优结果：")
        print(f"最大总遮蔽时长: {results['optimal_duration']:.3f} 秒")
        
        print(f"\n无人机飞行策略：")
        params = results['optimal_parameters']
        analysis = results['analysis']
        print(f"飞行方向角: {params['flight_angle']:.3f} 弧度 "
              f"({analysis['flight_direction_degrees']:.1f}°)")
        print(f"飞行速度: {params['flight_speed']:.1f} 米/秒")
        
        print(f"\n干扰弹部署策略：")
        for grenade in analysis['grenade_deployments']:
            print(f"干扰弹 {grenade['grenade_id']}:")
            print(f"  投放时间: {grenade['drop_time']:.2f} 秒")
            print(f"  起爆延迟: {grenade['detonation_delay']:.2f} 秒") 
            print(f"  起爆时刻: {grenade['detonation_time']:.2f} 秒")
            print(f"  起爆高度: {grenade['detonation_position'][2]:.1f} 米")
            print(f"  到真目标距离: {grenade['cloud_to_target_distance']:.1f} 米")
        
        print(f"\n时序分析：")
        timing = analysis['timing_summary']
        print(f"首次投放: {timing['first_drop']:.2f} 秒")
        print(f"最后投放: {timing['last_drop']:.2f} 秒")
        print(f"投放时间跨度: {timing['last_drop'] - timing['first_drop']:.2f} 秒")
        print(f"首次起爆: {timing['first_detonation']:.2f} 秒") 
        print(f"最后起爆: {timing['last_detonation']:.2f} 秒")
        print(f"最后烟幕消散: {timing['last_cloud_fade']:.2f} 秒")
        
        print(f"\n优化统计：")
        stats = results['optimization_stats']
        print(f"总评估次数: {stats['total_evaluations']}")
        print(f"收敛成功: {stats['convergence_success']}")

def main():
    """Main function to run Problem 3 solution"""
    print("正在求解问题3：单无人机多干扰弹部署策略...")
    
    # Create solver with 3 grenades
    solver = Problem3Solver(num_grenades=3, simulation_time_step=0.02)
    
    # Solve optimization problem
    results = solver.solve(max_iterations=100, population_size=18)
    
    # Print results
    solver.print_solution(results)
    
    return results

if __name__ == "__main__":
    results = main()