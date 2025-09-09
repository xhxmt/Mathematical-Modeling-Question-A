"""
Core physics and geometry calculations for smoke interference strategy optimization.

This module implements the fundamental mathematical models for:
- Trajectory calculations for missiles, drones, and grenades
- Geometric obscuration analysis using line-sphere intersection
- Time-based simulation of multiple moving objects
"""

import numpy as np
from typing import Tuple, List, Optional, Union
import warnings

class PhysicsConstants:
    """Physical constants and default parameters"""
    GRAVITY = np.array([0, 0, -9.8])  # m/s^2
    CLOUD_RADIUS = 10.0  # m
    CLOUD_SINK_VELOCITY = np.array([0, 0, -3.0])  # m/s
    CLOUD_LIFETIME = 20.0  # seconds
    MISSILE_SPEED = 300.0  # m/s
    DRONE_SPEED_MIN = 70.0  # m/s
    DRONE_SPEED_MAX = 140.0  # m/s

class Vector3D:
    """3D vector operations with optimization for trajectory calculations"""
    
    @staticmethod
    def normalize(vector: np.ndarray) -> np.ndarray:
        """Normalize vector to unit length"""
        norm = np.linalg.norm(vector)
        return vector / norm if norm > 1e-10 else np.zeros_like(vector)
    
    @staticmethod
    def distance(p1: np.ndarray, p2: np.ndarray) -> float:
        """Calculate distance between two points"""
        return np.linalg.norm(p2 - p1)
    
    @staticmethod
    def cross_product_magnitude(v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate magnitude of cross product"""
        return np.linalg.norm(np.cross(v1, v2))

class TrajectoryCalculator:
    """Calculate trajectories for missiles, drones, and projectiles"""
    
    @staticmethod
    def missile_position(initial_pos: np.ndarray, target_pos: np.ndarray, 
                        speed: float, time: float) -> np.ndarray:
        """
        Calculate missile position at given time (uniform linear motion)
        
        Args:
            initial_pos: Initial missile position [x, y, z]
            target_pos: Target position [x, y, z] 
            speed: Missile speed (m/s)
            time: Time elapsed (s)
            
        Returns:
            Position vector at time t
        """
        direction = Vector3D.normalize(target_pos - initial_pos)
        velocity = direction * speed
        return initial_pos + velocity * time
    
    @staticmethod
    def drone_position(initial_pos: np.ndarray, velocity: np.ndarray, 
                      time: float) -> np.ndarray:
        """
        Calculate drone position at given time (constant altitude flight)
        
        Args:
            initial_pos: Initial drone position [x, y, z]
            velocity: Drone velocity vector [vx, vy, 0]
            time: Time elapsed (s)
            
        Returns:
            Position vector at time t
        """
        return initial_pos + velocity * time
    
    @staticmethod
    def grenade_trajectory(drop_position: np.ndarray, initial_velocity: np.ndarray, 
                          time_since_drop: float) -> np.ndarray:
        """
        Calculate grenade position during projectile motion
        
        Args:
            drop_position: Position where grenade was dropped [x, y, z]
            initial_velocity: Initial velocity inherited from drone [vx, vy, vz]
            time_since_drop: Time since grenade was dropped (s)
            
        Returns:
            Position vector at time t after drop
        """
        if time_since_drop < 0:
            return drop_position
            
        gravity_effect = 0.5 * PhysicsConstants.GRAVITY * time_since_drop**2
        return drop_position + initial_velocity * time_since_drop + gravity_effect
    
    @staticmethod
    def smoke_cloud_center(detonation_pos: np.ndarray, 
                          time_since_detonation: float) -> np.ndarray:
        """
        Calculate smoke cloud center position (uniform sinking motion)
        
        Args:
            detonation_pos: Position where grenade detonated [x, y, z]
            time_since_detonation: Time since detonation (s)
            
        Returns:
            Cloud center position at time t after detonation
        """
        if time_since_detonation < 0:
            return detonation_pos
            
        return (detonation_pos + 
                PhysicsConstants.CLOUD_SINK_VELOCITY * time_since_detonation)

class ObscurationAnalyzer:
    """Analyze line-of-sight obscuration by spherical smoke clouds"""
    
    @staticmethod
    def check_line_sphere_intersection(line_start: np.ndarray, line_end: np.ndarray,
                                     sphere_center: np.ndarray, 
                                     sphere_radius: float) -> bool:
        """
        Check if line segment intersects with sphere
        
        Uses geometric method: distance from sphere center to line < radius
        AND intersection point is between line endpoints
        
        Args:
            line_start: Start point of line segment [x, y, z]
            line_end: End point of line segment [x, y, z]  
            sphere_center: Center of sphere [x, y, z]
            sphere_radius: Radius of sphere
            
        Returns:
            True if line intersects sphere, False otherwise
        """
        # Vector from start to end of line
        line_vector = line_end - line_start
        line_length = np.linalg.norm(line_vector)
        
        if line_length < 1e-10:  # Degenerate line
            return Vector3D.distance(line_start, sphere_center) <= sphere_radius
        
        # Vector from line start to sphere center
        start_to_center = sphere_center - line_start
        
        # Project sphere center onto line
        projection_scalar = np.dot(start_to_center, line_vector) / (line_length**2)
        
        # Check if projection is within line segment
        if projection_scalar < 0 or projection_scalar > 1:
            # Closest point is at one of the endpoints
            dist_to_start = Vector3D.distance(line_start, sphere_center)
            dist_to_end = Vector3D.distance(line_end, sphere_center)
            return min(dist_to_start, dist_to_end) <= sphere_radius
        
        # Closest point is on the line segment
        closest_point = line_start + projection_scalar * line_vector
        distance_to_line = Vector3D.distance(closest_point, sphere_center)
        
        return distance_to_line <= sphere_radius
    
    @staticmethod
    def is_target_obscured(missile_pos: np.ndarray, target_points: List[np.ndarray],
                          cloud_center: np.ndarray, cloud_radius: float) -> bool:
        """
        Check if target is obscured from missile view by smoke cloud
        
        Args:
            missile_pos: Current missile position [x, y, z]
            target_points: List of key points on target to check
            cloud_center: Center of smoke cloud [x, y, z]
            cloud_radius: Radius of smoke cloud
            
        Returns:
            True if at least one line of sight is blocked, False otherwise
        """
        for target_point in target_points:
            if ObscurationAnalyzer.check_line_sphere_intersection(
                missile_pos, target_point, cloud_center, cloud_radius):
                return True
        return False

class SimulationEngine:
    """Time-based simulation engine for multi-object trajectory analysis"""
    
    def __init__(self, time_step: float = 0.01):
        """
        Initialize simulation engine
        
        Args:
            time_step: Simulation time step in seconds
        """
        self.time_step = time_step
        self.current_time = 0.0
        
    def reset(self):
        """Reset simulation to initial state"""
        self.current_time = 0.0
        
    def calculate_obscuration_duration(self, 
                                     missile_initial_pos: np.ndarray,
                                     missile_target_pos: np.ndarray,
                                     drone_initial_pos: np.ndarray,
                                     drone_velocity: np.ndarray,
                                     drop_time: float,
                                     detonation_delay: float,
                                     target_points: List[np.ndarray],
                                     max_simulation_time: float = 100.0) -> float:
        """
        Calculate total obscuration duration for a single grenade scenario
        
        Args:
            missile_initial_pos: Initial missile position
            missile_target_pos: Missile target position (fake target origin)
            drone_initial_pos: Initial drone position  
            drone_velocity: Drone velocity vector
            drop_time: Time when grenade is dropped
            detonation_delay: Delay between drop and detonation
            target_points: Key points on real target to protect
            max_simulation_time: Maximum simulation time
            
        Returns:
            Total obscuration duration in seconds
        """
        self.reset()
        
        detonation_time = drop_time + detonation_delay
        cloud_fade_time = detonation_time + PhysicsConstants.CLOUD_LIFETIME
        
        # Calculate key positions
        drop_position = TrajectoryCalculator.drone_position(
            drone_initial_pos, drone_velocity, drop_time)
        
        detonation_position = TrajectoryCalculator.grenade_trajectory(
            drop_position, drone_velocity, detonation_delay)
        
        total_obscuration_time = 0.0
        
        # Simulate from detonation until cloud fades
        simulation_time = min(max_simulation_time, cloud_fade_time)
        
        for t in np.arange(detonation_time, simulation_time, self.time_step):
            # Calculate current positions
            missile_pos = TrajectoryCalculator.missile_position(
                missile_initial_pos, missile_target_pos, 
                PhysicsConstants.MISSILE_SPEED, t)
                
            cloud_center = TrajectoryCalculator.smoke_cloud_center(
                detonation_position, t - detonation_time)
            
            # Check obscuration
            if ObscurationAnalyzer.is_target_obscured(
                missile_pos, target_points, cloud_center, 
                PhysicsConstants.CLOUD_RADIUS):
                total_obscuration_time += self.time_step
                
        return total_obscuration_time