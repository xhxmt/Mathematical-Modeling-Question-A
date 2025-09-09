"""
Problem 5 Fixed: Multiple drones with multiple grenades against multiple missiles.

This is an improved version that fixes the zero obscuration duration issue.
"""

import numpy as np
from typing import List, Dict, Any, Tuple
from scipy.optimize import differential_evolution
import warnings
from .physics_core import (
    PhysicsConstants, TrajectoryCalculator, ObscurationAnalyzer, 
    SimulationEngine, Vector3D
)

class Problem5SolverFixed:
    """Fixed solver for Problem 5: Multi-Drone Multi-Grenade vs Multi-Missile"""
    
    def __init__(self, num_drones: int = 3, grenades_per_drone: int = 2, 
                 num_missiles: int = 3, simulation_time_step: float = 0.04):
        """
        Initialize Problem 5 solver with fixes
        
        Args:
            num_drones: Number of drones
            grenades_per_drone: Number of grenades per drone (reduced for stability)
            num_missiles: Number of missiles to defend against
            simulation_time_step: Time step for simulation (increased for speed)
        """
        self.num_drones = num_drones
        self.grenades_per_drone = grenades_per_drone
        self.num_missiles = num_missiles
        self.simulation_engine = SimulationEngine(simulation_time_step)
        
        # Missile configurations (same as before but verified)
        self.missile_configs = [
            {
                'missile_id': 1,
                'initial_position': np.array([20000.0, 0.0, 2000.0]),
                'target_position': np.array([0.0, 200.0, 5.0]),  # Real target, not fake!
                'speed': PhysicsConstants.MISSILE_SPEED
            },
            {
                'missile_id': 2,
                'initial_position': np.array([19000.0, -1000.0, 2100.0]),
                'target_position': np.array([0.0, 200.0, 5.0]),  # Real target
                'speed': PhysicsConstants.MISSILE_SPEED
            },
            {
                'missile_id': 3,
                'initial_position': np.array([19000.0, 1000.0, 1900.0]),
                'target_position': np.array([0.0, 200.0, 5.0]),   # Real target
                'speed': PhysicsConstants.MISSILE_SPEED
            }
        ]
        
        # Drone initial positions (improved positioning)
        self.drone_initial_positions = [
            np.array([15000.0, 0.0, 1500.0]),     # FY1 - closer to action
            np.array([15000.0, -1000.0, 1500.0]), # FY2 - spread out
            np.array([15000.0, 1000.0, 1500.0])   # FY3 - spread out
        ]
        
        # Target definition
        self.real_target_center = np.array([0.0, 200.0, 5.0])
        self.real_target_radius = 200.0
        self.real_target_height = 10.0
        self.target_key_points = self._generate_target_key_points()
        
        # Improved decision variable bounds
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
        
        # Simplified key points for better performance
        points.append(center + np.array([0, 0, height/2]))     # Top center
        points.append(center + np.array([0, 0, -height/2]))    # Bottom center
        points.append(center + np.array([radius/2, 0, 0]))     # Side point 1
        points.append(center + np.array([-radius/2, 0, 0]))    # Side point 2
        points.append(center + np.array([0, radius/2, 0]))     # Side point 3
        points.append(center + np.array([0, -radius/2, 0]))    # Side point 4
            
        return points
    
    def _setup_optimization_bounds(self) -> List[Tuple[float, float]]:
        """Setup improved optimization bounds for all decision variables"""
        bounds = []
        
        # For each drone - more focused parameter ranges
        for i in range(self.num_drones):
            # Drone flight parameters - focus on effective directions
            bounds.extend([
                (0.0, np.pi),          # flight_angle (forward hemisphere only)
                (90.0, 140.0),         # flight_speed (higher speeds for better positioning)
            ])
            
            # Grenade deployment parameters - more realistic timing
            for j in range(self.grenades_per_drone):
                bounds.extend([
                    (0.5, 20.0),       # drop_time (earlier deployment)
                    (1.0, 6.0)         # detonation_delay (shorter, more practical)
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
            
            # Sort and enforce minimum drop intervals (more lenient)
            grenades.sort(key=lambda g: g['drop_time'])
            for j in range(1, len(grenades)):
                min_time = grenades[j-1]['drop_time'] + 0.5  # Reduced interval
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
    
    def calculate_total_obscuration_simplified(self, drone_strategies: List[Dict]) -> Dict[str, float]:
        """
        Simplified and more robust obscuration calculation
        """
        results = {}
        
        # Calculate all cloud deployments
        all_clouds = []
        for strategy in drone_strategies:
            for grenade in strategy['grenades']:
                drop_time = grenade['drop_time']
                detonation_delay = grenade['detonation_delay']
                detonation_time = drop_time + detonation_delay
                
                # Calculate positions
                drop_position = TrajectoryCalculator.drone_position(
                    strategy['initial_position'], strategy['velocity'], drop_time)
                detonation_position = TrajectoryCalculator.grenade_trajectory(
                    drop_position, strategy['velocity'], detonation_delay)
                
                all_clouds.append({
                    'drone_id': strategy['drone_id'],
                    'grenade_id': grenade['grenade_id'],
                    'detonation_time': detonation_time,
                    'detonation_position': detonation_position,
                    'fade_time': detonation_time + PhysicsConstants.CLOUD_LIFETIME
                })
        
        if not all_clouds:
            return {f'missile_{i+1}': 0.0 for i in range(self.num_missiles)}
        
        # Calculate obscuration for each missile
        total_obscuration = 0.0
        
        for missile_config in self.missile_configs:
            missile_id = missile_config['missile_id']
            missile_obscuration = 0.0
            
            # Find time window for this missile
            earliest_detonation = min(cloud['detonation_time'] for cloud in all_clouds)
            latest_fade = max(cloud['fade_time'] for cloud in all_clouds)
            
            # Simulate in time steps
            time_step = self.simulation_engine.time_step
            
            for t in np.arange(earliest_detonation, min(latest_fade, 100.0), time_step):
                # Calculate missile position
                missile_pos = TrajectoryCalculator.missile_position(
                    missile_config['initial_position'],
                    missile_config['target_position'],
                    missile_config['speed'], t)
                
                # Check if any cloud provides obscuration
                is_obscured = False
                
                for cloud in all_clouds:
                    if cloud['detonation_time'] <= t <= cloud['fade_time']:
                        time_since_detonation = t - cloud['detonation_time']
                        cloud_center = TrajectoryCalculator.smoke_cloud_center(
                            cloud['detonation_position'], time_since_detonation)
                        
                        # Check obscuration for any target point
                        for target_point in self.target_key_points:
                            if ObscurationAnalyzer.check_line_sphere_intersection(
                                missile_pos, target_point, cloud_center, 
                                PhysicsConstants.CLOUD_RADIUS):
                                is_obscured = True
                                break
                    
                    if is_obscured:
                        break
                
                if is_obscured:
                    missile_obscuration += time_step
            
            results[f'missile_{missile_id}'] = missile_obscuration
            total_obscuration += missile_obscuration
        
        results['total_obscuration'] = total_obscuration
        results['average_obscuration'] = total_obscuration / self.num_missiles if self.num_missiles > 0 else 0.0
        
        return results
    
    def objective_function(self, x: np.ndarray) -> float:
        """
        Improved objective function with better error handling and debugging
        """
        self.evaluation_count += 1
        
        try:
            params = self._decode_decision_variables(x)
            
            # Calculate obscuration results
            obscuration_results = self.calculate_total_obscuration_simplified(
                params['drone_strategies'])
            
            # Simple objective: maximize total obscuration
            total_obscuration = obscuration_results['total_obscuration']
            
            # Add small bonus for using more grenades effectively
            num_active_grenades = len([g for strategy in params['drone_strategies'] 
                                     for g in strategy['grenades'] if g['drop_time'] < 30])
            bonus = 0.01 * num_active_grenades
            
            objective_value = total_obscuration + bonus
            
            # Store good solutions
            if objective_value > 0:
                self.best_solutions.append({
                    'parameters': params.copy(),
                    'obscuration_results': obscuration_results.copy(),
                    'objective_value': objective_value,
                    'evaluation': self.evaluation_count
                })
                
                # Print progress for better solutions
                if len(self.best_solutions) <= 3 or total_obscuration > max(s['obscuration_results']['total_obscuration'] for s in self.best_solutions[:-1]):
                    print(f"    评估 {self.evaluation_count}: 发现解 {total_obscuration:.3f}秒")
            
            return -objective_value  # Negative for minimization
            
        except Exception as e:
            print(f"    评估错误 {self.evaluation_count}: {e}")
            return 0.0
    
    def solve(self, max_iterations: int = 40, population_size: int = 15) -> Dict[str, Any]:
        """
        Solve Problem 5 using a more conservative approach
        """
        print(f"修复版问题5求解器：{self.num_drones}架无人机×{self.grenades_per_drone}枚干扰弹 对 {self.num_missiles}枚导弹")
        print(f"决策变量维度: {len(self.bounds)}")
        print(f"使用保守参数: {max_iterations}次迭代, 种群大小{population_size}")
        
        # Reset optimization tracking
        self.evaluation_count = 0
        self.best_solutions = []
        
        # Run optimization with more conservative settings
        result = differential_evolution(
            self.objective_function,
            self.bounds,
            strategy='best1bin',
            maxiter=max_iterations,
            popsize=population_size,
            seed=42,
            disp=False,  # Reduce verbose output
            atol=1e-4,
            tol=1e-4,
            mutation=(0.5, 1.2),
            recombination=0.7
        )
        
        # Process results
        optimal_params = self._decode_decision_variables(result.x)
        
        # Re-evaluate optimal solution for detailed analysis
        optimal_obscuration = self.calculate_total_obscuration_simplified(
            optimal_params['drone_strategies'])
        
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
                'best_solutions_found': len(self.best_solutions),
                'version': 'fixed'
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
                    [(i+1, obscuration_results.get(f'missile_{i+1}', 0)) 
                     for i in range(self.num_missiles)],
                    key=lambda x: x[1]) if obscuration_results else (None, 0)
            }
        }

# Export the fixed solver
Problem5Solver = Problem5SolverFixed