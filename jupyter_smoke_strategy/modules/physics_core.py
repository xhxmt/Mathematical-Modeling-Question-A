"""
Enhanced physics core module for Jupyter-based smoke strategy optimization.

This module provides improved physics calculations with better visualization
support and interactive analysis capabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import pandas as pd

@dataclass
class PhysicsConfig:
    """Configuration class for physics constants"""
    GRAVITY: np.ndarray = np.array([0, 0, -9.8])  # m/s^2
    CLOUD_RADIUS: float = 10.0  # m
    CLOUD_SINK_VELOCITY: np.ndarray = np.array([0, 0, -3.0])  # m/s
    CLOUD_LIFETIME: float = 20.0  # seconds
    MISSILE_SPEED: float = 300.0  # m/s
    DRONE_SPEED_MIN: float = 70.0  # m/s
    DRONE_SPEED_MAX: float = 140.0  # m/s
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for easy display"""
        return {
            'Gravity (m/s²)': self.GRAVITY,
            'Cloud Radius (m)': self.CLOUD_RADIUS,
            'Cloud Sink Speed (m/s)': abs(self.CLOUD_SINK_VELOCITY[2]),
            'Cloud Lifetime (s)': self.CLOUD_LIFETIME,
            'Missile Speed (m/s)': self.MISSILE_SPEED,
            'Drone Speed Range (m/s)': f"{self.DRONE_SPEED_MIN}-{self.DRONE_SPEED_MAX}"
        }

class EnhancedTrajectoryCalculator:
    """Enhanced trajectory calculator with visualization support"""
    
    @staticmethod
    def missile_trajectory_array(initial_pos: np.ndarray, target_pos: np.ndarray, 
                               speed: float, time_array: np.ndarray) -> np.ndarray:
        """
        Calculate missile trajectory for array of time points (vectorized)
        
        Args:
            initial_pos: Initial missile position [x, y, z]
            target_pos: Target position [x, y, z]
            speed: Missile speed (m/s)
            time_array: Array of time points
            
        Returns:
            Array of positions [N, 3] where N is len(time_array)
        """
        direction = target_pos - initial_pos
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-10:
            return np.tile(initial_pos, (len(time_array), 1))
        
        unit_direction = direction / direction_norm
        velocity = unit_direction * speed
        
        # Vectorized calculation
        positions = initial_pos[np.newaxis, :] + velocity[np.newaxis, :] * time_array[:, np.newaxis]
        return positions
    
    @staticmethod
    def drone_trajectory_array(initial_pos: np.ndarray, velocity: np.ndarray, 
                             time_array: np.ndarray) -> np.ndarray:
        """
        Calculate drone trajectory for array of time points (vectorized)
        
        Args:
            initial_pos: Initial drone position [x, y, z]
            velocity: Drone velocity vector [vx, vy, 0]
            time_array: Array of time points
            
        Returns:
            Array of positions [N, 3]
        """
        positions = initial_pos[np.newaxis, :] + velocity[np.newaxis, :] * time_array[:, np.newaxis]
        return positions
    
    @staticmethod
    def grenade_trajectory_array(drop_position: np.ndarray, initial_velocity: np.ndarray, 
                               time_array: np.ndarray, 
                               config: PhysicsConfig = PhysicsConfig()) -> np.ndarray:
        """
        Calculate grenade trajectory for array of time points (vectorized)
        
        Args:
            drop_position: Position where grenade was dropped [x, y, z]
            initial_velocity: Initial velocity inherited from drone [vx, vy, vz]
            time_array: Array of time points since drop
            config: Physics configuration
            
        Returns:
            Array of positions [N, 3]
        """
        # Handle negative times
        time_array = np.maximum(time_array, 0)
        
        # Vectorized projectile motion calculation
        positions = (drop_position[np.newaxis, :] + 
                    initial_velocity[np.newaxis, :] * time_array[:, np.newaxis] +
                    0.5 * config.GRAVITY[np.newaxis, :] * time_array[:, np.newaxis]**2)
        
        return positions

class EnhancedObscurationAnalyzer:
    """Enhanced obscuration analyzer with batch processing capabilities"""
    
    @staticmethod
    def batch_line_sphere_intersection(line_starts: np.ndarray, line_ends: np.ndarray,
                                     sphere_center: np.ndarray, 
                                     sphere_radius: float) -> np.ndarray:
        """
        Batch check if multiple line segments intersect with sphere (vectorized)
        
        Args:
            line_starts: Start points of line segments [N, 3]
            line_ends: End points of line segments [N, 3]  
            sphere_center: Center of sphere [3,]
            sphere_radius: Radius of sphere
            
        Returns:
            Boolean array [N,] indicating intersection for each line
        """
        N = len(line_starts)
        
        # Vector from start to end of each line
        line_vectors = line_ends - line_starts
        line_lengths = np.linalg.norm(line_vectors, axis=1)
        
        # Handle degenerate lines
        degenerate_mask = line_lengths < 1e-10
        results = np.zeros(N, dtype=bool)
        
        if np.any(degenerate_mask):
            distances = np.linalg.norm(line_starts[degenerate_mask] - sphere_center, axis=1)
            results[degenerate_mask] = distances <= sphere_radius
        
        # Process non-degenerate lines
        valid_mask = ~degenerate_mask
        if not np.any(valid_mask):
            return results
        
        valid_starts = line_starts[valid_mask]
        valid_vectors = line_vectors[valid_mask]
        valid_lengths = line_lengths[valid_mask]
        
        # Vector from line start to sphere center
        start_to_center = sphere_center[np.newaxis, :] - valid_starts
        
        # Project sphere center onto each line
        projections = np.sum(start_to_center * valid_vectors, axis=1) / (valid_lengths**2)
        
        # Check if projection is within line segment
        within_segment = (projections >= 0) & (projections <= 1)
        
        # Calculate distances
        distances = np.full(len(valid_starts), np.inf)
        
        # For projections outside segment, use endpoint distances
        outside_mask = ~within_segment
        if np.any(outside_mask):
            dist_to_start = np.linalg.norm(start_to_center[outside_mask], axis=1)
            dist_to_end = np.linalg.norm(
                valid_starts[outside_mask] + valid_vectors[outside_mask] - sphere_center, axis=1)
            distances[outside_mask] = np.minimum(dist_to_start, dist_to_end)
        
        # For projections within segment, calculate perpendicular distance
        inside_mask = within_segment
        if np.any(inside_mask):
            closest_points = (valid_starts[inside_mask] + 
                            projections[inside_mask, np.newaxis] * valid_vectors[inside_mask])
            distances[inside_mask] = np.linalg.norm(closest_points - sphere_center, axis=1)
        
        # Check intersection
        results[valid_mask] = distances <= sphere_radius
        
        return results
    
    @staticmethod
    def batch_target_obscuration(missile_positions: np.ndarray, target_points: List[np.ndarray],
                               cloud_center: np.ndarray, cloud_radius: float) -> np.ndarray:
        """
        Check target obscuration for multiple missile positions (vectorized)
        
        Args:
            missile_positions: Missile positions [N, 3]
            target_points: List of key points on target
            cloud_center: Center of smoke cloud [3,]
            cloud_radius: Radius of smoke cloud
            
        Returns:
            Boolean array [N,] indicating obscuration for each position
        """
        N = len(missile_positions)
        obscured = np.zeros(N, dtype=bool)
        
        for target_point in target_points:
            target_array = np.tile(target_point, (N, 1))
            intersections = EnhancedObscurationAnalyzer.batch_line_sphere_intersection(
                missile_positions, target_array, cloud_center, cloud_radius)
            obscured |= intersections
            
        return obscured

class InteractiveSimulationEngine:
    """Enhanced simulation engine with interactive capabilities"""
    
    def __init__(self, time_step: float = 0.02, config: PhysicsConfig = PhysicsConfig()):
        self.time_step = time_step
        self.config = config
        self.simulation_data = {}
        
    def reset(self):
        """Reset simulation data"""
        self.simulation_data = {}
        
    def simulate_scenario(self, scenario_params: Dict[str, Any], 
                         max_time: float = 100.0) -> Dict[str, Any]:
        """
        Simulate complete scenario with detailed trajectory data
        
        Args:
            scenario_params: Dictionary containing all scenario parameters
            max_time: Maximum simulation time
            
        Returns:
            Complete simulation results with trajectories
        """
        self.reset()
        
        # Time array for simulation
        time_array = np.arange(0, max_time, self.time_step)
        
        # Extract parameters
        missile_params = scenario_params['missiles']
        drone_params = scenario_params['drones']
        target_points = scenario_params['target_points']
        
        # Calculate all trajectories
        missile_trajectories = {}
        for missile_id, missile in missile_params.items():
            trajectory = EnhancedTrajectoryCalculator.missile_trajectory_array(
                missile['initial_pos'], missile['target_pos'], 
                missile['speed'], time_array)
            missile_trajectories[missile_id] = trajectory
        
        drone_trajectories = {}
        for drone_id, drone in drone_params.items():
            trajectory = EnhancedTrajectoryCalculator.drone_trajectory_array(
                drone['initial_pos'], drone['velocity'], time_array)
            drone_trajectories[drone_id] = trajectory
        
        # Calculate grenade trajectories and cloud periods
        cloud_data = []
        for drone_id, drone in drone_params.items():
            for grenade in drone.get('grenades', []):
                drop_time = grenade['drop_time']
                detonation_delay = grenade['detonation_delay']
                detonation_time = drop_time + detonation_delay
                
                # Grenade trajectory
                drop_idx = int(drop_time / self.time_step)
                if drop_idx < len(time_array):
                    drop_position = drone_trajectories[drone_id][drop_idx]
                    
                    grenade_time_array = time_array - drop_time
                    grenade_trajectory = EnhancedTrajectoryCalculator.grenade_trajectory_array(
                        drop_position, drone['velocity'], grenade_time_array, self.config)
                    
                    # Find detonation position
                    det_idx = int(detonation_time / self.time_step) if detonation_time < max_time else -1
                    if det_idx >= 0 and det_idx < len(grenade_trajectory):
                        detonation_position = grenade_trajectory[det_idx]
                        
                        cloud_data.append({
                            'drone_id': drone_id,
                            'grenade_id': grenade['grenade_id'],
                            'drop_time': drop_time,
                            'detonation_time': detonation_time,
                            'detonation_position': detonation_position,
                            'fade_time': detonation_time + self.config.CLOUD_LIFETIME,
                            'trajectory': grenade_trajectory
                        })
        
        # Calculate obscuration for each missile
        obscuration_results = {}
        for missile_id, trajectory in missile_trajectories.items():
            obscuration_timeline = np.zeros(len(time_array), dtype=bool)
            
            for cloud in cloud_data:
                # Find time indices when this cloud is active
                start_idx = max(0, int(cloud['detonation_time'] / self.time_step))
                end_idx = min(len(time_array), int(cloud['fade_time'] / self.time_step))
                
                if start_idx < end_idx:
                    # Calculate cloud positions during active period
                    active_times = time_array[start_idx:end_idx]
                    cloud_positions = []
                    
                    for t in active_times:
                        time_since_det = t - cloud['detonation_time']
                        cloud_pos = (cloud['detonation_position'] + 
                                   self.config.CLOUD_SINK_VELOCITY * time_since_det)
                        cloud_positions.append(cloud_pos)
                    
                    # Check obscuration for this period
                    if cloud_positions:
                        for i, cloud_pos in enumerate(cloud_positions):
                            missile_pos = trajectory[start_idx + i]
                            is_obscured = EnhancedObscurationAnalyzer.batch_target_obscuration(
                                missile_pos[np.newaxis, :], target_points, 
                                cloud_pos, self.config.CLOUD_RADIUS)[0]
                            obscuration_timeline[start_idx + i] |= is_obscured
            
            # Calculate total obscuration duration
            total_duration = np.sum(obscuration_timeline) * self.time_step
            obscuration_results[missile_id] = {
                'total_duration': total_duration,
                'timeline': obscuration_timeline,
                'percentage': 100 * total_duration / max_time
            }
        
        # Store simulation data
        self.simulation_data = {
            'time_array': time_array,
            'missile_trajectories': missile_trajectories,
            'drone_trajectories': drone_trajectories,
            'cloud_data': cloud_data,
            'obscuration_results': obscuration_results,
            'scenario_params': scenario_params
        }
        
        return self.simulation_data
    
    def get_simulation_summary(self) -> pd.DataFrame:
        """Get simulation results as a pandas DataFrame"""
        if not self.simulation_data:
            return pd.DataFrame()
        
        results = []
        for missile_id, result in self.simulation_data['obscuration_results'].items():
            results.append({
                'Missile': missile_id,
                'Total Duration (s)': result['total_duration'],
                'Percentage (%)': result['percentage']
            })
        
        return pd.DataFrame(results)

def create_default_scenario() -> Dict[str, Any]:
    """Create a default scenario for testing and demonstration"""
    return {
        'missiles': {
            'M1': {
                'initial_pos': np.array([20000.0, 0.0, 2000.0]),
                'target_pos': np.array([0.0, 200.0, 5.0]),
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
            np.array([0.0, 200.0, 10.0]),  # Top
            np.array([0.0, 200.0, 0.0]),   # Bottom
            np.array([200.0, 200.0, 5.0]), # Side
            np.array([-200.0, 200.0, 5.0]) # Side
        ]
    }

# Export main classes and functions
__all__ = [
    'PhysicsConfig',
    'EnhancedTrajectoryCalculator', 
    'EnhancedObscurationAnalyzer',
    'InteractiveSimulationEngine',
    'create_default_scenario'
]