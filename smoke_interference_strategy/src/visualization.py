"""
3D Visualization module for smoke interference strategy optimization.

This module provides comprehensive 3D visualization capabilities to display:
- Missile, drone, and grenade trajectories
- Smoke cloud evolution and obscuration analysis
- Real-time animation of the complete scenario
- Multi-scenario comparison views
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from typing import List, Dict, Any, Optional, Tuple
import warnings

from .physics_core import (
    PhysicsConstants, TrajectoryCalculator, Vector3D
)

class Visualization3D:
    """3D visualization engine for trajectory and obscuration analysis"""
    
    def __init__(self, figsize: Tuple[int, int] = (15, 10)):
        """
        Initialize 3D visualization engine
        
        Args:
            figsize: Figure size for plots
        """
        self.figsize = figsize
        self.colors = {
            'missile': 'red',
            'drone': 'blue', 
            'grenade': 'orange',
            'smoke_cloud': 'gray',
            'target_real': 'green',
            'target_fake': 'black',
            'trajectory_missile': 'darkred',
            'trajectory_drone': 'darkblue',
            'trajectory_grenade': 'darkorange'
        }
        
    def plot_static_scenario(self, scenario_data: Dict[str, Any], 
                           title: str = "Smoke Interference Strategy Analysis") -> plt.Figure:
        """
        Create static 3D plot of complete scenario
        
        Args:
            scenario_data: Dictionary containing all trajectory and position data
            title: Plot title
            
        Returns:
            Matplotlib figure object
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot target positions
        self._plot_targets(ax, scenario_data)
        
        # Plot trajectories
        self._plot_trajectories(ax, scenario_data)
        
        # Plot key event positions
        self._plot_key_positions(ax, scenario_data)
        
        # Plot smoke clouds at detonation moments
        self._plot_smoke_clouds(ax, scenario_data)
        
        # Configure plot
        self._configure_3d_plot(ax, title)
        
        plt.tight_layout()
        return fig
    
    def create_trajectory_animation(self, scenario_data: Dict[str, Any],
                                  animation_duration: float = 60.0,
                                  fps: int = 20) -> FuncAnimation:
        """
        Create animated visualization of trajectory evolution
        
        Args:
            scenario_data: Scenario data dictionary
            animation_duration: Total animation duration in seconds
            fps: Frames per second
            
        Returns:
            Matplotlib animation object
        """
        fig = plt.figure(figsize=self.figsize)
        ax = fig.add_subplot(111, projection='3d')
        
        # Setup animation parameters
        total_frames = int(animation_duration * fps)
        time_step = animation_duration / total_frames
        
        # Initialize plot elements
        self._setup_animation_elements(ax, scenario_data)
        
        def animate(frame):
            current_time = frame * time_step
            self._update_animation_frame(ax, scenario_data, current_time)
            return []
        
        animation = FuncAnimation(fig, animate, frames=total_frames, 
                                interval=1000/fps, blit=False, repeat=True)
        
        return animation
    
    def plot_multi_scenario_comparison(self, scenarios: List[Dict[str, Any]],
                                     scenario_names: List[str]) -> plt.Figure:
        """
        Create comparison plot for multiple scenarios
        
        Args:
            scenarios: List of scenario data dictionaries
            scenario_names: Names for each scenario
            
        Returns:
            Matplotlib figure with subplots
        """
        n_scenarios = len(scenarios)
        cols = min(3, n_scenarios)
        rows = (n_scenarios + cols - 1) // cols
        
        fig = plt.figure(figsize=(cols * 5, rows * 4))
        
        for i, (scenario, name) in enumerate(zip(scenarios, scenario_names)):
            ax = fig.add_subplot(rows, cols, i + 1, projection='3d')
            
            # Plot scenario
            self._plot_targets(ax, scenario)
            self._plot_trajectories(ax, scenario)
            self._plot_key_positions(ax, scenario)
            self._plot_smoke_clouds(ax, scenario)
            
            # Configure subplot
            self._configure_3d_plot(ax, name, compact=True)
        
        plt.tight_layout()
        return fig
    
    def plot_obscuration_timeline(self, obscuration_data: Dict[str, Any]) -> plt.Figure:
        """
        Plot timeline showing obscuration periods
        
        Args:
            obscuration_data: Dictionary with time series obscuration data
            
        Returns:
            Matplotlib figure
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Timeline plot
        if 'timeline' in obscuration_data:
            times = obscuration_data['timeline']['times']
            obscured = obscuration_data['timeline']['obscured']
            
            ax1.fill_between(times, 0, obscured, alpha=0.3, color='red', 
                           label='Target Obscured')
            ax1.plot(times, obscured, 'r-', linewidth=1)
            ax1.set_ylabel('Obscuration Status')
            ax1.set_title('Target Obscuration Timeline')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Cloud activity timeline
        if 'clouds' in obscuration_data:
            for i, cloud in enumerate(obscuration_data['clouds']):
                start_time = cloud['detonation_time']
                end_time = cloud['fade_time']
                ax2.barh(i, end_time - start_time, left=start_time, 
                        alpha=0.6, label=f"Cloud {i+1}")
            
            ax2.set_xlabel('Time (seconds)')
            ax2.set_ylabel('Smoke Cloud ID')
            ax2.set_title('Smoke Cloud Activity Timeline')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig
    
    def _plot_targets(self, ax, scenario_data: Dict[str, Any]):
        """Plot real and fake targets"""
        # Real target (cylinder)
        if 'real_target' in scenario_data:
            target = scenario_data['real_target']
            center = target['center']
            radius = target['radius']
            height = target['height']
            
            # Draw cylinder representation
            theta = np.linspace(0, 2*np.pi, 20)
            z_bottom = center[2] - height/2
            z_top = center[2] + height/2
            
            # Bottom circle
            x_circle = center[0] + radius * np.cos(theta)
            y_circle = center[1] + radius * np.sin(theta)
            z_circle = np.full_like(x_circle, z_bottom)
            ax.plot(x_circle, y_circle, z_circle, 'g-', linewidth=2, label='Real Target Base')
            
            # Top circle
            z_circle_top = np.full_like(x_circle, z_top)
            ax.plot(x_circle, y_circle, z_circle_top, 'g-', linewidth=2, label='Real Target Top')
            
            # Vertical lines
            for i in range(0, len(theta), 4):
                ax.plot([x_circle[i], x_circle[i]], 
                       [y_circle[i], y_circle[i]], 
                       [z_bottom, z_top], 'g-', alpha=0.5)
        
        # Fake target
        if 'fake_target' in scenario_data:
            fake_pos = scenario_data['fake_target']
            ax.scatter(*fake_pos, color='black', s=100, marker='x', 
                      linewidth=3, label='Fake Target')
    
    def _plot_trajectories(self, ax, scenario_data: Dict[str, Any]):
        """Plot missile, drone, and grenade trajectories"""
        # Missile trajectories
        if 'missile_trajectories' in scenario_data:
            for i, trajectory in enumerate(scenario_data['missile_trajectories']):
                positions = trajectory['positions']
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], 
                       color=self.colors['trajectory_missile'], alpha=0.7,
                       linewidth=2, label=f'Missile M{i+1}' if i == 0 else "")
        
        # Drone trajectories  
        if 'drone_trajectories' in scenario_data:
            for i, trajectory in enumerate(scenario_data['drone_trajectories']):
                positions = trajectory['positions']
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                       color=self.colors['trajectory_drone'], alpha=0.7,
                       linewidth=2, label=f'Drone FY{i+1}' if i == 0 else "")
        
        # Grenade trajectories
        if 'grenade_trajectories' in scenario_data:
            for i, trajectory in enumerate(scenario_data['grenade_trajectories']):
                positions = trajectory['positions']
                ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
                       color=self.colors['trajectory_grenade'], alpha=0.6,
                       linewidth=1, linestyle='--', 
                       label=f'Grenade {i+1}' if i == 0 else "")
    
    def _plot_key_positions(self, ax, scenario_data: Dict[str, Any]):
        """Plot key event positions (drop, detonation, etc.)"""
        # Drop positions
        if 'drop_positions' in scenario_data:
            for i, pos in enumerate(scenario_data['drop_positions']):
                ax.scatter(*pos, color='orange', s=80, marker='o', 
                          alpha=0.8, label='Drop Point' if i == 0 else "")
        
        # Detonation positions
        if 'detonation_positions' in scenario_data:
            for i, pos in enumerate(scenario_data['detonation_positions']):
                ax.scatter(*pos, color='red', s=100, marker='*', 
                          alpha=0.9, label='Detonation Point' if i == 0 else "")
    
    def _plot_smoke_clouds(self, ax, scenario_data: Dict[str, Any]):
        """Plot smoke cloud spheres at detonation positions"""
        if 'smoke_clouds' in scenario_data:
            for i, cloud in enumerate(scenario_data['smoke_clouds']):
                center = cloud['center']
                radius = cloud.get('radius', PhysicsConstants.CLOUD_RADIUS)
                
                # Create sphere wireframe
                u = np.linspace(0, 2 * np.pi, 10)
                v = np.linspace(0, np.pi, 10)
                x_sphere = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y_sphere = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z_sphere = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                
                ax.plot_wireframe(x_sphere, y_sphere, z_sphere, alpha=0.3, 
                                color=self.colors['smoke_cloud'])
    
    def _configure_3d_plot(self, ax, title: str, compact: bool = False):
        """Configure 3D plot appearance and labels"""
        ax.set_xlabel('X (meters)', fontsize=10 if compact else 12)
        ax.set_ylabel('Y (meters)', fontsize=10 if compact else 12) 
        ax.set_zlabel('Z (meters)', fontsize=10 if compact else 12)
        ax.set_title(title, fontsize=12 if compact else 14)
        
        # Set equal aspect ratio
        self._set_equal_aspect_3d(ax)
        
        # Add legend if not compact
        if not compact:
            ax.legend(loc='upper left', bbox_to_anchor=(0, 1))
        
        # Grid
        ax.grid(True, alpha=0.3)
        
        # Set view angle
        ax.view_init(elev=20, azim=45)
    
    def _set_equal_aspect_3d(self, ax):
        """Set equal aspect ratio for 3D plot"""
        # Get current limits
        xlim = ax.get_xlim3d()
        ylim = ax.get_ylim3d()
        zlim = ax.get_zlim3d()
        
        # Calculate ranges
        xrange = xlim[1] - xlim[0]
        yrange = ylim[1] - ylim[0]
        zrange = zlim[1] - zlim[0]
        
        # Set equal ranges
        max_range = max(xrange, yrange, zrange)
        
        # Center points
        xcenter = (xlim[1] + xlim[0]) / 2
        ycenter = (ylim[1] + ylim[0]) / 2
        zcenter = (zlim[1] + zlim[0]) / 2
        
        # Set new limits
        ax.set_xlim3d(xcenter - max_range/2, xcenter + max_range/2)
        ax.set_ylim3d(ycenter - max_range/2, ycenter + max_range/2)
        ax.set_zlim3d(max(0, zcenter - max_range/2), zcenter + max_range/2)
    
    def _setup_animation_elements(self, ax, scenario_data: Dict[str, Any]):
        """Setup elements for animation"""
        # Plot static elements
        self._plot_targets(ax, scenario_data)
        self._configure_3d_plot(ax, "Smoke Interference Strategy Animation")
    
    def _update_animation_frame(self, ax, scenario_data: Dict[str, Any], current_time: float):
        """Update animation frame for given time"""
        ax.clear()
        
        # Redraw static elements
        self._plot_targets(ax, scenario_data)
        
        # Calculate and plot current positions
        self._plot_current_positions(ax, scenario_data, current_time)
        
        # Reconfigure plot
        self._configure_3d_plot(ax, f"Time: {current_time:.1f}s")
    
    def _plot_current_positions(self, ax, scenario_data: Dict[str, Any], current_time: float):
        """Plot current positions of all objects at given time"""
        # This would need to be implemented based on specific trajectory data
        # For now, we'll plot static trajectories
        self._plot_trajectories(ax, scenario_data)
        self._plot_key_positions(ax, scenario_data)

def create_scenario_data_from_results(problem_results: Dict[str, Any], 
                                    problem_type: str = "problem1") -> Dict[str, Any]:
    """
    Convert problem solver results to visualization data format
    
    Args:
        problem_results: Results from problem solver
        problem_type: Type of problem (problem1, problem2, etc.)
        
    Returns:
        Scenario data dictionary for visualization
    """
    scenario_data = {
        'real_target': {
            'center': np.array([0.0, 200.0, 5.0]),
            'radius': 200.0,
            'height': 10.0
        },
        'fake_target': np.array([0.0, 0.0, 0.0])
    }
    
    # Add missile trajectory
    missile_initial = np.array([20000.0, 0.0, 2000.0])
    missile_target = np.array([0.0, 0.0, 0.0])
    
    # Generate missile trajectory points
    times = np.linspace(0, 60, 100)  # 60 seconds
    missile_positions = []
    for t in times:
        pos = TrajectoryCalculator.missile_position(
            missile_initial, missile_target, PhysicsConstants.MISSILE_SPEED, t)
        missile_positions.append(pos)
    
    scenario_data['missile_trajectories'] = [{
        'positions': np.array(missile_positions)
    }]
    
    # Add problem-specific data
    if problem_type in ["problem1", "problem2"] and 'strategy_parameters' in problem_results:
        # Single drone scenario
        params = problem_results['strategy_parameters']
        drone_initial = params['drone_initial_position']
        drone_velocity = params['drone_velocity']
        
        # Generate drone trajectory
        drone_positions = []
        for t in times:
            pos = TrajectoryCalculator.drone_position(drone_initial, drone_velocity, t)
            drone_positions.append(pos)
        
        scenario_data['drone_trajectories'] = [{
            'positions': np.array(drone_positions)
        }]
        
        # Add key positions
        if 'key_positions' in problem_results:
            positions = problem_results['key_positions']
            scenario_data['drop_positions'] = [positions['drop_position']]
            scenario_data['detonation_positions'] = [positions['detonation_position']]
            scenario_data['smoke_clouds'] = [{
                'center': positions['detonation_position'],
                'radius': PhysicsConstants.CLOUD_RADIUS
            }]
    
    return scenario_data

def save_visualization_plots(results: Dict[str, Any], problem_name: str, 
                           output_dir: str = "results/"):
    """
    Save visualization plots to files
    
    Args:
        results: Problem solver results
        problem_name: Name of the problem (e.g., "problem1")
        output_dir: Output directory for saved plots
    """
    try:
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Create visualization engine
        viz = Visualization3D()
        
        # Convert results to scenario data
        scenario_data = create_scenario_data_from_results(results, problem_name)
        
        # Create and save static plot
        fig = viz.plot_static_scenario(scenario_data, f"{problem_name.title()} Solution")
        fig.savefig(f"{output_dir}{problem_name}_trajectory.png", dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Visualization saved to {output_dir}{problem_name}_trajectory.png")
        
    except Exception as e:
        warnings.warn(f"Could not save visualization: {e}")

if __name__ == "__main__":
    # Test visualization with dummy data
    viz = Visualization3D()
    
    # Create sample scenario data
    scenario_data = {
        'real_target': {
            'center': np.array([0.0, 200.0, 5.0]),
            'radius': 200.0,
            'height': 10.0
        },
        'fake_target': np.array([0.0, 0.0, 0.0]),
        'missile_trajectories': [{
            'positions': np.array([[20000, 0, 2000], [10000, 0, 1000], [0, 0, 0]])
        }],
        'drone_trajectories': [{
            'positions': np.array([[17800, 0, 1800], [10000, 0, 1800], [2000, 0, 1800]])
        }],
        'drop_positions': [np.array([15000, 0, 1800])],
        'detonation_positions': [np.array([12000, 0, 1600])],
        'smoke_clouds': [{
            'center': np.array([12000, 0, 1600]),
            'radius': 10.0
        }]
    }
    
    # Create plot
    fig = viz.plot_static_scenario(scenario_data, "Test Scenario")
    plt.show()
    print("Visualization test completed.")