"""
Problem 1: Calculate obscuration duration for fixed deployment strategy.

This module implements the solution for the first problem: given fixed parameters
for drone flight and grenade deployment, calculate the effective obscuration duration
against missile M1.
"""

import numpy as np
from typing import List, Dict, Any
from .physics_core import (
    PhysicsConstants, TrajectoryCalculator, ObscurationAnalyzer, 
    SimulationEngine, Vector3D
)

class Problem1Solver:
    """Solver for Problem 1: Fixed Strategy Obscuration Duration"""
    
    def __init__(self, simulation_time_step: float = 0.01):
        """
        Initialize Problem 1 solver
        
        Args:
            simulation_time_step: Time step for numerical simulation
        """
        self.simulation_engine = SimulationEngine(simulation_time_step)
        
        # Fixed parameters from problem statement
        self.missile_initial_pos = np.array([20000.0, 0.0, 2000.0])  # M1 initial position
        self.drone_initial_pos = np.array([17800.0, 0.0, 1800.0])   # FY1 initial position
        self.fake_target_pos = np.array([0.0, 0.0, 0.0])            # Fake target at origin
        
        # Real target definition (cylinder: radius=200m, height=10m, center at (0,200,5))
        self.real_target_center = np.array([0.0, 200.0, 5.0])
        self.real_target_radius = 200.0
        self.real_target_height = 10.0
        
        # Generate key points on real target for obscuration analysis
        self.target_key_points = self._generate_target_key_points()
        
        # Fixed strategy parameters
        self.drone_speed = 120.0  # m/s, towards fake target
        self.drop_time = 1.5      # seconds after mission start
        self.detonation_delay = 3.6  # seconds after drop
        
    def _generate_target_key_points(self) -> List[np.ndarray]:
        """
        Generate key points on the cylindrical real target for obscuration analysis
        
        Returns:
            List of 3D points representing critical target locations
        """
        points = []
        center = self.real_target_center
        radius = self.real_target_radius
        height = self.real_target_height
        
        # Center points of top and bottom circular faces
        points.append(center + np.array([0, 0, height/2]))   # Top center
        points.append(center + np.array([0, 0, -height/2]))  # Bottom center
        
        # Points on the circumference of top and bottom faces
        angles = np.linspace(0, 2*np.pi, 8, endpoint=False)  # 8 points around circumference
        
        for angle in angles:
            x_offset = radius * np.cos(angle)
            y_offset = radius * np.sin(angle)
            
            # Top circumference
            points.append(center + np.array([x_offset, y_offset, height/2]))
            # Bottom circumference  
            points.append(center + np.array([x_offset, y_offset, -height/2]))
            
        return points
    
    def _calculate_drone_velocity(self) -> np.ndarray:
        """
        Calculate drone velocity vector based on fixed strategy
        
        Returns:
            3D velocity vector for drone (z-component is 0 for constant altitude)
        """
        # Direction from drone initial position to fake target (2D projection)
        drone_pos_2d = self.drone_initial_pos[:2]  # [x, y] only
        target_pos_2d = self.fake_target_pos[:2]   # [x, y] only
        
        direction_2d = Vector3D.normalize(target_pos_2d - drone_pos_2d)
        
        # Create 3D velocity vector with zero z-component
        velocity_3d = np.array([
            direction_2d[0] * self.drone_speed,
            direction_2d[1] * self.drone_speed,
            0.0  # Constant altitude flight
        ])
        
        return velocity_3d
    
    def solve(self) -> Dict[str, Any]:
        """
        Solve Problem 1: calculate obscuration duration for fixed strategy
        
        Returns:
            Dictionary containing solution results and analysis
        """
        # Calculate drone velocity vector
        drone_velocity = self._calculate_drone_velocity()
        
        # Calculate obscuration duration using simulation engine
        obscuration_duration = self.simulation_engine.calculate_obscuration_duration(
            missile_initial_pos=self.missile_initial_pos,
            missile_target_pos=self.fake_target_pos,
            drone_initial_pos=self.drone_initial_pos,
            drone_velocity=drone_velocity,
            drop_time=self.drop_time,
            detonation_delay=self.detonation_delay,
            target_points=self.target_key_points
        )
        
        # Calculate additional analysis parameters
        detonation_time = self.drop_time + self.detonation_delay
        
        # Positions at key moments
        drop_position = TrajectoryCalculator.drone_position(
            self.drone_initial_pos, drone_velocity, self.drop_time)
            
        detonation_position = TrajectoryCalculator.grenade_trajectory(
            drop_position, drone_velocity, self.detonation_delay)
            
        missile_pos_at_detonation = TrajectoryCalculator.missile_position(
            self.missile_initial_pos, self.fake_target_pos,
            PhysicsConstants.MISSILE_SPEED, detonation_time)
        
        # Distance analysis
        missile_to_target_distance = Vector3D.distance(
            self.missile_initial_pos, self.real_target_center)
        cloud_to_target_distance = Vector3D.distance(
            detonation_position, self.real_target_center)
        
        # Compile results
        results = {
            'obscuration_duration': obscuration_duration,
            'strategy_parameters': {
                'drone_initial_position': self.drone_initial_pos,
                'drone_velocity': drone_velocity,
                'drone_speed': self.drone_speed,
                'drop_time': self.drop_time,
                'detonation_delay': self.detonation_delay,
                'detonation_time': detonation_time
            },
            'key_positions': {
                'drop_position': drop_position,
                'detonation_position': detonation_position,
                'missile_position_at_detonation': missile_pos_at_detonation
            },
            'distance_analysis': {
                'missile_to_target_initial': missile_to_target_distance,
                'cloud_to_target': cloud_to_target_distance,
                'cloud_altitude': detonation_position[2]
            },
            'target_definition': {
                'real_target_center': self.real_target_center,
                'real_target_radius': self.real_target_radius,
                'real_target_height': self.real_target_height,
                'key_points_count': len(self.target_key_points)
            }
        }
        
        return results
    
    def print_solution(self, results: Dict[str, Any]) -> None:
        """
        Print formatted solution results
        
        Args:
            results: Results dictionary from solve() method
        """
        print("=" * 60)
        print("问题1解决方案：固定策略下的遮蔽时长计算")
        print("=" * 60)
        
        print(f"\n主要结果：")
        print(f"有效遮蔽时长: {results['obscuration_duration']:.3f} 秒")
        
        print(f"\n策略参数：")
        params = results['strategy_parameters']
        print(f"无人机初始位置: ({params['drone_initial_position'][0]:.1f}, "
              f"{params['drone_initial_position'][1]:.1f}, "
              f"{params['drone_initial_position'][2]:.1f}) 米")
        print(f"无人机飞行速度: {params['drone_speed']:.1f} 米/秒")
        print(f"投放时间: {params['drop_time']:.1f} 秒")
        print(f"起爆延迟: {params['detonation_delay']:.1f} 秒")
        print(f"起爆时刻: {params['detonation_time']:.1f} 秒")
        
        print(f"\n关键位置：")
        positions = results['key_positions']
        print(f"投放点位置: ({positions['drop_position'][0]:.1f}, "
              f"{positions['drop_position'][1]:.1f}, "
              f"{positions['drop_position'][2]:.1f}) 米")
        print(f"起爆点位置: ({positions['detonation_position'][0]:.1f}, "
              f"{positions['detonation_position'][1]:.1f}, "
              f"{positions['detonation_position'][2]:.1f}) 米")
        
        print(f"\n几何分析：")
        distances = results['distance_analysis']
        print(f"导弹到真目标初始距离: {distances['missile_to_target_initial']:.1f} 米")
        print(f"烟幕云团到真目标距离: {distances['cloud_to_target']:.1f} 米") 
        print(f"烟幕云团起爆高度: {distances['cloud_altitude']:.1f} 米")

def main():
    """Main function to run Problem 1 solution"""
    print("正在求解问题1：固定策略下的遮蔽时长计算...")
    
    # Create solver and run analysis
    solver = Problem1Solver(simulation_time_step=0.01)
    results = solver.solve()
    
    # Print results
    solver.print_solution(results)
    
    return results

if __name__ == "__main__":
    results = main()