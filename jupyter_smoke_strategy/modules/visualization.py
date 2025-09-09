"""
Advanced visualization module for Jupyter-based smoke strategy optimization.

This module provides interactive 3D visualizations, animated plots, and
comprehensive analysis charts optimized for Jupyter notebooks.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import seaborn as sns
import pandas as pd
from typing import List, Dict, Any, Optional, Tuple
import ipywidgets as widgets
from IPython.display import display, HTML
import warnings

# Set style for better notebook display
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class InteractiveVisualizer:
    """Interactive 3D visualizer for smoke strategy scenarios"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = {
            'missile': '#FF4444',
            'drone': '#4444FF', 
            'grenade': '#FF8800',
            'smoke_cloud': '#888888',
            'target_real': '#44FF44',
            'target_fake': '#000000',
            'trajectory_missile': '#CC0000',
            'trajectory_drone': '#0000CC',
            'trajectory_grenade': '#CC6600'
        }
    
    def plot_3d_scenario_plotly(self, simulation_data: Dict[str, Any], 
                               title: str = "Smoke Strategy 3D Visualization") -> go.Figure:
        """
        Create interactive 3D visualization using Plotly
        
        Args:
            simulation_data: Complete simulation data
            title: Plot title
            
        Returns:
            Plotly figure object
        """
        fig = go.Figure()
        
        # Plot missile trajectories
        for missile_id, trajectory in simulation_data['missile_trajectories'].items():
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
                mode='lines',
                name=f'Missile {missile_id}',
                line=dict(color=self.colors['trajectory_missile'], width=4),
                hovertemplate="<b>%{fullData.name}</b><br>" +
                            "X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<br>" +
                            "<extra></extra>"
            ))
        
        # Plot drone trajectories  
        for drone_id, trajectory in simulation_data['drone_trajectories'].items():
            fig.add_trace(go.Scatter3d(
                x=trajectory[:, 0], y=trajectory[:, 1], z=trajectory[:, 2],
                mode='lines',
                name=f'Drone {drone_id}',
                line=dict(color=self.colors['trajectory_drone'], width=3),
                hovertemplate="<b>%{fullData.name}</b><br>" +
                            "X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<br>" +
                            "<extra></extra>"
            ))
        
        # Plot grenade trajectories and detonation points
        for cloud in simulation_data['cloud_data']:
            trajectory = cloud['trajectory']
            
            # Grenade trajectory (until detonation)
            detonation_time = cloud['detonation_time']
            time_array = simulation_data['time_array']
            detonation_idx = int(detonation_time / (time_array[1] - time_array[0]))
            
            if detonation_idx < len(trajectory):
                active_trajectory = trajectory[:detonation_idx+1]
                
                fig.add_trace(go.Scatter3d(
                    x=active_trajectory[:, 0], 
                    y=active_trajectory[:, 1], 
                    z=active_trajectory[:, 2],
                    mode='lines',
                    name=f'Grenade {cloud["drone_id"]}-{cloud["grenade_id"]}',
                    line=dict(color=self.colors['trajectory_grenade'], width=2, dash='dash'),
                    hovertemplate="<b>%{fullData.name}</b><br>" +
                                "X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<br>" +
                                "<extra></extra>"
                ))
                
                # Detonation point
                det_pos = cloud['detonation_position']
                fig.add_trace(go.Scatter3d(
                    x=[det_pos[0]], y=[det_pos[1]], z=[det_pos[2]],
                    mode='markers',
                    name=f'Detonation {cloud["drone_id"]}-{cloud["grenade_id"]}',
                    marker=dict(
                        size=8,
                        color='red',
                        symbol='diamond',
                    ),
                    hovertemplate="<b>Detonation Point</b><br>" +
                                f"Time: {detonation_time:.1f}s<br>" +
                                "X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<br>" +
                                "<extra></extra>"
                ))
        
        # Plot target area
        target_points = simulation_data['scenario_params']['target_points']
        if target_points:
            target_array = np.array(target_points)
            fig.add_trace(go.Scatter3d(
                x=target_array[:, 0], 
                y=target_array[:, 1], 
                z=target_array[:, 2],
                mode='markers',
                name='Target Points',
                marker=dict(
                    size=6,
                    color=self.colors['target_real'],
                    symbol='square',
                ),
                hovertemplate="<b>Target Point</b><br>" +
                            "X: %{x:.0f}m<br>Y: %{y:.0f}m<br>Z: %{z:.0f}m<br>" +
                            "<extra></extra>"
            ))
        
        # Update layout
        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X (meters)",
                yaxis_title="Y (meters)", 
                zaxis_title="Z (meters)",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.2)
                )
            ),
            width=800,
            height=600,
            showlegend=True
        )
        
        return fig
    
    def plot_obscuration_timeline(self, simulation_data: Dict[str, Any]) -> go.Figure:
        """
        Plot obscuration timeline for all missiles
        
        Args:
            simulation_data: Complete simulation data
            
        Returns:
            Plotly figure with timeline visualization
        """
        fig = make_subplots(
            rows=len(simulation_data['obscuration_results']),
            cols=1,
            subplot_titles=[f"Missile {mid}" for mid in simulation_data['obscuration_results'].keys()],
            shared_xaxes=True,
            vertical_spacing=0.05
        )
        
        time_array = simulation_data['time_array']
        
        for i, (missile_id, result) in enumerate(simulation_data['obscuration_results'].items()):
            timeline = result['timeline'].astype(int)  # Convert bool to int for plotting
            
            fig.add_trace(
                go.Scatter(
                    x=time_array,
                    y=timeline,
                    mode='lines',
                    name=f'Missile {missile_id}',
                    line=dict(color=self.colors['missile'], width=2),
                    fill='tozeroy',
                    fillcolor=f'rgba(255, 68, 68, 0.3)',
                    hovertemplate=f"<b>Missile {missile_id}</b><br>" +
                                "Time: %{x:.1f}s<br>" +
                                "Obscured: %{y}<br>" +
                                "<extra></extra>"
                ),
                row=i+1, col=1
            )
            
            # Add cloud activation periods
            for cloud in simulation_data['cloud_data']:
                fig.add_vrect(
                    x0=cloud['detonation_time'],
                    x1=cloud['fade_time'],
                    fillcolor="rgba(128, 128, 128, 0.2)",
                    layer="below",
                    line_width=0,
                    row=i+1, col=1
                )
        
        fig.update_layout(
            title="Target Obscuration Timeline",
            height=200 * len(simulation_data['obscuration_results']),
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time (seconds)", row=len(simulation_data['obscuration_results']), col=1)
        fig.update_yaxes(title_text="Obscured", range=[-0.1, 1.1])
        
        return fig
    
    def create_parameter_dashboard(self, simulation_engine) -> widgets.VBox:
        """
        Create interactive parameter dashboard
        
        Args:
            simulation_engine: Simulation engine instance
            
        Returns:
            IPython widget for parameter control
        """
        # Parameter controls
        drone_speed = widgets.FloatSlider(
            value=120.0, min=70.0, max=140.0, step=5.0,
            description='Drone Speed:', style={'description_width': 'initial'}
        )
        
        drop_time = widgets.FloatSlider(
            value=1.5, min=0.1, max=10.0, step=0.1,
            description='Drop Time:', style={'description_width': 'initial'}
        )
        
        detonation_delay = widgets.FloatSlider(
            value=3.6, min=0.1, max=8.0, step=0.1,
            description='Detonation Delay:', style={'description_width': 'initial'}
        )
        
        flight_angle = widgets.FloatSlider(
            value=180.0, min=0.0, max=360.0, step=5.0,
            description='Flight Angle:', style={'description_width': 'initial'}
        )
        
        # Output area
        output = widgets.Output()
        
        def update_simulation(*args):
            with output:
                output.clear_output(wait=True)
                
                # Create scenario with current parameters
                scenario = {
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
                            'velocity': np.array([
                                drone_speed.value * np.cos(np.radians(flight_angle.value)),
                                drone_speed.value * np.sin(np.radians(flight_angle.value)),
                                0.0
                            ]),
                            'grenades': [
                                {'grenade_id': 1, 'drop_time': drop_time.value, 
                                 'detonation_delay': detonation_delay.value}
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
                
                # Run simulation
                results = simulation_engine.simulate_scenario(scenario, max_time=60.0)
                
                # Display results
                summary = simulation_engine.get_simulation_summary()
                print("Simulation Results:")
                print(summary.to_string(index=False))
                
                # Create visualization
                fig = self.plot_3d_scenario_plotly(results, "Interactive Scenario")
                fig.show()
        
        # Connect sliders to update function
        for slider in [drone_speed, drop_time, detonation_delay, flight_angle]:
            slider.observe(update_simulation, names='value')
        
        # Initial update
        update_simulation()
        
        # Create layout
        controls = widgets.VBox([
            widgets.HTML("<h3>Interactive Parameter Control</h3>"),
            drone_speed, drop_time, detonation_delay, flight_angle,
            widgets.HTML("<hr>")
        ])
        
        return widgets.VBox([controls, output])

class OptimizationVisualizer:
    """Visualizer for optimization process and results"""
    
    @staticmethod
    def plot_optimization_history(optimization_history: List[Dict]) -> go.Figure:
        """
        Plot optimization convergence history
        
        Args:
            optimization_history: List of optimization results over iterations
            
        Returns:
            Plotly figure showing convergence
        """
        if not optimization_history:
            return go.Figure()
        
        df = pd.DataFrame(optimization_history)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=["Best Objective Value", "Population Statistics"],
            shared_xaxes=True
        )
        
        # Best value over time
        fig.add_trace(
            go.Scatter(
                x=df['iteration'],
                y=df['best_value'],
                mode='lines+markers',
                name='Best Value',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Population statistics
        if 'mean_value' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['iteration'],
                    y=df['mean_value'],
                    mode='lines',
                    name='Mean Value',
                    line=dict(color='orange')
                ),
                row=2, col=1
            )
        
        if 'std_value' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df['iteration'],
                    y=df['std_value'],
                    mode='lines',
                    name='Standard Deviation',
                    line=dict(color='red')
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            title="Optimization Convergence",
            height=600
        )
        
        fig.update_xaxes(title_text="Iteration", row=2, col=1)
        fig.update_yaxes(title_text="Objective Value", row=1, col=1)
        fig.update_yaxes(title_text="Value", row=2, col=1)
        
        return fig
    
    @staticmethod
    def plot_parameter_sensitivity(results_data: List[Dict], param_name: str) -> go.Figure:
        """
        Plot parameter sensitivity analysis
        
        Args:
            results_data: List of results with different parameter values
            param_name: Name of the parameter being analyzed
            
        Returns:
            Plotly figure showing sensitivity
        """
        df = pd.DataFrame(results_data)
        
        fig = px.scatter(
            df, x=param_name, y='objective_value',
            title=f"Sensitivity Analysis: {param_name}",
            labels={param_name: param_name.replace('_', ' ').title(),
                   'objective_value': 'Objective Value'}
        )
        
        fig.add_trace(
            go.Scatter(
                x=df[param_name],
                y=df['objective_value'],
                mode='lines',
                name='Trend',
                line=dict(color='red', dash='dash')
            )
        )
        
        return fig

class ComparisonVisualizer:
    """Visualizer for comparing different strategies and solutions"""
    
    @staticmethod
    def plot_strategy_comparison(strategies: List[Dict[str, Any]]) -> go.Figure:
        """
        Compare multiple strategies side by side
        
        Args:
            strategies: List of strategy results to compare
            
        Returns:
            Plotly figure with comparison
        """
        if not strategies:
            return go.Figure()
        
        # Prepare data
        strategy_names = [s['name'] for s in strategies]
        total_durations = [s['results']['total_duration'] for s in strategies]
        
        # Create comparison chart
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=strategy_names,
            y=total_durations,
            name='Total Obscuration Duration',
            marker_color='lightblue',
            text=[f"{d:.2f}s" for d in total_durations],
            textposition='auto'
        ))
        
        fig.update_layout(
            title="Strategy Performance Comparison",
            xaxis_title="Strategy",
            yaxis_title="Total Obscuration Duration (seconds)",
            height=400
        )
        
        return fig
    
    @staticmethod
    def create_results_table(strategies: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Create detailed comparison table
        
        Args:
            strategies: List of strategy results
            
        Returns:
            DataFrame with comparison data
        """
        if not strategies:
            return pd.DataFrame()
        
        results = []
        for strategy in strategies:
            result = {
                'Strategy': strategy['name'],
                'Total Duration (s)': strategy['results']['total_duration'],
                'Computation Time (s)': strategy.get('computation_time', 0),
                'Efficiency (s/s)': strategy['results']['total_duration'] / max(strategy.get('computation_time', 1), 1)
            }
            results.append(result)
        
        return pd.DataFrame(results).round(3)

# Utility functions
def setup_notebook_display():
    """Setup notebook for optimal display of visualizations"""
    display(HTML("""
    <style>
    .widget-label { min-width: 150px !important; }
    .widget-readout { min-width: 80px !important; }
    </style>
    """))
    
    # Enable plotly in notebook
    try:
        import plotly.offline as pyo
        pyo.init_notebook_mode(connected=True)
    except:
        pass

# Export main classes
__all__ = [
    'InteractiveVisualizer',
    'OptimizationVisualizer', 
    'ComparisonVisualizer',
    'setup_notebook_display'
]