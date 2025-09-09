"""
Utility functions for Jupyter-based smoke strategy optimization.

This module provides helper functions, data processing tools, result analysis,
and common utilities used across the project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, List, Any, Tuple, Optional, Union
import json
import pickle
import os
from datetime import datetime
import warnings

def save_results(results: Dict[str, Any], filename: str, 
                format: str = 'json', compress: bool = False) -> None:
    """
    Save results to file with various format options
    
    Args:
        results: Results dictionary to save
        filename: Output filename
        format: Save format ('json', 'pickle', 'csv')
        compress: Whether to compress the file
    """
    
    # Ensure data directory exists
    os.makedirs(os.path.dirname(filename) if os.path.dirname(filename) else '.', exist_ok=True)
    
    if format.lower() == 'json':
        def convert_numpy(obj):
            """Convert numpy arrays to lists for JSON serialization"""
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            elif isinstance(obj, tuple):
                return tuple(convert_numpy(item) for item in obj)
            return obj
        
        converted_results = convert_numpy(results)
        
        if compress:
            import gzip
            with gzip.open(filename + '.gz', 'wt', encoding='utf-8') as f:
                json.dump(converted_results, f, indent=2, ensure_ascii=False)
        else:
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(converted_results, f, indent=2, ensure_ascii=False)
                
    elif format.lower() == 'pickle':
        if compress:
            import gzip
            with gzip.open(filename + '.gz', 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filename, 'wb') as f:
                pickle.dump(results, f, protocol=pickle.HIGHEST_PROTOCOL)
                
    elif format.lower() == 'csv':
        # Convert to DataFrame if possible
        if isinstance(results, dict) and all(isinstance(v, (list, np.ndarray)) for v in results.values()):
            df = pd.DataFrame(results)
            if compress:
                df.to_csv(filename + '.gz', compression='gzip', index=False)
            else:
                df.to_csv(filename, index=False)
        else:
            warnings.warn("Cannot save complex nested dictionary as CSV")
            
    else:
        raise ValueError(f"Unsupported format: {format}")

def load_results(filename: str, format: str = 'auto') -> Dict[str, Any]:
    """
    Load results from file
    
    Args:
        filename: Input filename  
        format: Load format ('json', 'pickle', 'csv', 'auto')
        
    Returns:
        Loaded results dictionary
    """
    
    if format == 'auto':
        # Auto-detect format from extension
        if filename.endswith('.json') or filename.endswith('.json.gz'):
            format = 'json'
        elif filename.endswith('.pkl') or filename.endswith('.pickle') or filename.endswith('.pkl.gz'):
            format = 'pickle'
        elif filename.endswith('.csv') or filename.endswith('.csv.gz'):
            format = 'csv'
        else:
            raise ValueError(f"Cannot auto-detect format for {filename}")
    
    compressed = filename.endswith('.gz')
    
    if format.lower() == 'json':
        if compressed:
            import gzip
            with gzip.open(filename, 'rt', encoding='utf-8') as f:
                return json.load(f)
        else:
            with open(filename, 'r', encoding='utf-8') as f:
                return json.load(f)
                
    elif format.lower() == 'pickle':
        if compressed:
            import gzip
            with gzip.open(filename, 'rb') as f:
                return pickle.load(f)
        else:
            with open(filename, 'rb') as f:
                return pickle.load(f)
                
    elif format.lower() == 'csv':
        if compressed:
            df = pd.read_csv(filename, compression='gzip')
        else:
            df = pd.read_csv(filename)
        return df.to_dict('list')
        
    else:
        raise ValueError(f"Unsupported format: {format}")

def create_results_summary(optimization_results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create a summary DataFrame from multiple optimization results
    
    Args:
        optimization_results: List of optimization result dictionaries
        
    Returns:
        Summary DataFrame
    """
    
    summary_data = []
    
    for i, result in enumerate(optimization_results):
        summary_row = {
            'Run': i + 1,
            'Problem': result.get('problem_name', 'Unknown'),
            'Algorithm': result.get('algorithm', 'Unknown'),
            'Best_Fitness': result.get('best_fitness', 0.0),
            'Total_Evaluations': result.get('total_evaluations', 0),
            'Computation_Time': result.get('computation_time', 0.0),
            'Converged': result.get('converged', False),
            'Final_Generation': result.get('final_generation', 0)
        }
        
        # Add solution parameters if available
        if 'best_solution' in result:
            solution = result['best_solution']
            if isinstance(solution, (list, np.ndarray)):
                for j, param in enumerate(solution):
                    summary_row[f'Param_{j+1}'] = param
                    
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)

def analyze_convergence(optimization_history: List[Dict], 
                       title: str = "Optimization Convergence") -> go.Figure:
    """
    Create convergence analysis plots
    
    Args:
        optimization_history: List of optimization history data
        title: Plot title
        
    Returns:
        Plotly figure with convergence plots
    """
    
    if not optimization_history:
        return go.Figure().add_annotation(text="No convergence data available", 
                                        showarrow=False)
    
    df = pd.DataFrame(optimization_history)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Best Fitness Over Time', 'Population Diversity', 
                       'Evaluation Efficiency', 'Convergence Rate'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": True}, {"secondary_y": False}]]
    )
    
    # Plot 1: Best fitness over generations
    fig.add_trace(
        go.Scatter(
            x=df.get('generation', df.get('iteration', range(len(df)))),
            y=df['best_fitness'],
            mode='lines+markers',
            name='Best Fitness',
            line=dict(color='blue', width=2)
        ),
        row=1, col=1
    )
    
    # Plot 2: Population diversity (standard deviation)
    if 'std_fitness' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.get('generation', df.get('iteration', range(len(df)))),
                y=df['std_fitness'],
                mode='lines',
                name='Population Std',
                line=dict(color='red', width=2)
            ),
            row=1, col=2
        )
    
    # Plot 3: Mean fitness and evaluation count
    if 'mean_fitness' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.get('generation', df.get('iteration', range(len(df)))),
                y=df['mean_fitness'],
                mode='lines',
                name='Mean Fitness',
                line=dict(color='green')
            ),
            row=2, col=1
        )
    
    if 'evaluations' in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df.get('generation', df.get('iteration', range(len(df)))),
                y=df['evaluations'],
                mode='lines',
                name='Total Evaluations',
                line=dict(color='orange', dash='dash'),
                yaxis='y2'
            ),
            row=2, col=1, secondary_y=True
        )
    
    # Plot 4: Convergence rate (improvement per generation)
    if len(df) > 1:
        improvements = np.diff(df['best_fitness'].values)
        fig.add_trace(
            go.Scatter(
                x=df.get('generation', df.get('iteration', range(len(df))))[1:],
                y=improvements,
                mode='lines+markers',
                name='Improvement Rate',
                line=dict(color='purple')
            ),
            row=2, col=2
        )
    
    # Update layout
    fig.update_layout(
        title=title,
        height=600,
        showlegend=True
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="Generation/Iteration")
    fig.update_yaxes(title_text="Fitness Value", row=1, col=1)
    fig.update_yaxes(title_text="Standard Deviation", row=1, col=2)
    fig.update_yaxes(title_text="Mean Fitness", row=2, col=1)
    fig.update_yaxes(title_text="Total Evaluations", row=2, col=1, secondary_y=True)
    fig.update_yaxes(title_text="Improvement", row=2, col=2)
    
    return fig

def compare_algorithms(results_list: List[Dict[str, Any]], 
                      metric: str = 'best_fitness') -> go.Figure:
    """
    Compare performance of different optimization algorithms
    
    Args:
        results_list: List of algorithm results
        metric: Metric to compare ('best_fitness', 'computation_time', etc.)
        
    Returns:
        Plotly figure with comparison
    """
    
    # Extract comparison data
    algorithms = []
    values = []
    colors = []
    
    color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    
    for i, result in enumerate(results_list):
        algorithm_name = result.get('algorithm', f'Algorithm_{i+1}')
        value = result.get(metric, 0)
        
        algorithms.append(algorithm_name)
        values.append(value)
        colors.append(color_palette[i % len(color_palette)])
    
    # Create bar chart
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(
            x=algorithms,
            y=values,
            marker_color=colors,
            text=[f'{v:.4f}' for v in values],
            textposition='auto'
        )
    )
    
    fig.update_layout(
        title=f"Algorithm Comparison: {metric.replace('_', ' ').title()}",
        xaxis_title="Algorithm",
        yaxis_title=metric.replace('_', ' ').title(),
        height=500
    )
    
    return fig

def parameter_sensitivity_heatmap(sensitivity_data: Dict[str, Any],
                                parameter_names: List[str] = None) -> go.Figure:
    """
    Create parameter sensitivity heatmap
    
    Args:
        sensitivity_data: Dictionary with sensitivity analysis results
        parameter_names: Names of parameters for axis labels
        
    Returns:
        Plotly heatmap figure
    """
    
    if 'results_matrix' in sensitivity_data:
        matrix = np.array(sensitivity_data['results_matrix'])
        
        # Get parameter ranges
        param1_range = sensitivity_data.get('param1_range', range(matrix.shape[0]))
        param2_range = sensitivity_data.get('param2_range', range(matrix.shape[1]))
        
        # Default parameter names
        if parameter_names is None:
            parameter_names = ['Parameter 1', 'Parameter 2']
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=param2_range,
            y=param1_range,
            colorscale='Viridis',
            colorbar=dict(title="Objective Value"),
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Parameter Sensitivity Analysis",
            xaxis_title=parameter_names[1] if len(parameter_names) > 1 else 'Parameter 2',
            yaxis_title=parameter_names[0] if len(parameter_names) > 0 else 'Parameter 1',
            width=800,
            height=600
        )
        
        return fig
    
    else:
        return go.Figure().add_annotation(text="No sensitivity matrix data available", 
                                        showarrow=False)

def export_results_report(results: Dict[str, Any], filename: str) -> None:
    """
    Export a comprehensive results report as HTML
    
    Args:
        results: Results dictionary
        filename: Output HTML filename
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Smoke Strategy Optimization Results</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; }}
            .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
            .section {{ margin: 20px 0; }}
            .metric {{ background-color: #e8f4f8; padding: 10px; margin: 5px 0; border-left: 4px solid #2196F3; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Smoke Strategy Optimization Results</h1>
            <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Problem Summary</h2>
            <div class="metric">Problem Type: {results.get('problem_name', 'Unknown')}</div>
            <div class="metric">Algorithm: {results.get('algorithm', 'Unknown')}</div>
            <div class="metric">Best Fitness: {results.get('best_fitness', 'N/A')}</div>
        </div>
        
        <div class="section">
            <h2>Performance Metrics</h2>
            <div class="metric">Total Evaluations: {results.get('total_evaluations', 'N/A')}</div>
            <div class="metric">Computation Time: {results.get('computation_time', 'N/A')} seconds</div>
            <div class="metric">Converged: {results.get('converged', 'N/A')}</div>
        </div>
        
        <div class="section">
            <h2>Optimal Solution</h2>
            <p>Best solution parameters:</p>
            <ul>
    """
    
    if 'best_solution' in results:
        solution = results['best_solution']
        if isinstance(solution, (list, np.ndarray)):
            for i, param in enumerate(solution):
                html_content += f"<li>Parameter {i+1}: {param:.6f}</li>"
    
    html_content += """
            </ul>
        </div>
    </body>
    </html>
    """
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write(html_content)

def calculate_performance_metrics(results: Dict[str, Any]) -> Dict[str, float]:
    """
    Calculate various performance metrics from optimization results
    
    Args:
        results: Optimization results dictionary
        
    Returns:
        Dictionary of performance metrics
    """
    
    metrics = {}
    
    # Basic metrics
    metrics['best_fitness'] = results.get('best_fitness', 0.0)
    metrics['total_evaluations'] = results.get('total_evaluations', 0)
    metrics['computation_time'] = results.get('computation_time', 0.0)
    
    # Efficiency metrics
    if metrics['computation_time'] > 0:
        metrics['evaluations_per_second'] = metrics['total_evaluations'] / metrics['computation_time']
        metrics['fitness_per_second'] = metrics['best_fitness'] / metrics['computation_time']
    
    # Convergence metrics
    history = results.get('convergence_history', [])
    if history:
        initial_fitness = history[0].get('best_fitness', 0.0)
        final_fitness = history[-1].get('best_fitness', 0.0)
        
        metrics['improvement_ratio'] = (final_fitness - initial_fitness) / max(abs(initial_fitness), 1e-10)
        metrics['convergence_generations'] = len(history)
        
        # Convergence rate (fitness improvement per generation)
        if len(history) > 1:
            fitness_values = [h.get('best_fitness', 0.0) for h in history]
            improvements = np.diff(fitness_values)
            metrics['average_improvement_rate'] = np.mean(improvements[improvements > 0]) if any(improvements > 0) else 0.0
            
    return metrics

def create_parameter_table(solution: Union[List, np.ndarray], 
                          parameter_names: List[str] = None,
                          bounds: List[Tuple[float, float]] = None) -> pd.DataFrame:
    """
    Create a formatted parameter table
    
    Args:
        solution: Solution vector
        parameter_names: Names of parameters
        bounds: Parameter bounds for validation
        
    Returns:
        Formatted DataFrame
    """
    
    if parameter_names is None:
        parameter_names = [f'Parameter_{i+1}' for i in range(len(solution))]
    
    data = {
        'Parameter': parameter_names,
        'Value': [f'{val:.6f}' for val in solution],
    }
    
    if bounds is not None:
        data['Lower_Bound'] = [bound[0] for bound in bounds]
        data['Upper_Bound'] = [bound[1] for bound in bounds]
        data['Within_Bounds'] = [
            '✓' if bounds[i][0] <= solution[i] <= bounds[i][1] else '✗'
            for i in range(len(solution))
        ]
    
    return pd.DataFrame(data)

def validate_solution(solution: Union[List, np.ndarray], 
                     bounds: List[Tuple[float, float]]) -> Dict[str, Any]:
    """
    Validate solution against bounds and constraints
    
    Args:
        solution: Solution to validate
        bounds: Parameter bounds
        
    Returns:
        Validation results dictionary
    """
    
    validation = {
        'is_valid': True,
        'bound_violations': [],
        'warnings': []
    }
    
    for i, (value, (lower, upper)) in enumerate(zip(solution, bounds)):
        if not (lower <= value <= upper):
            validation['is_valid'] = False
            validation['bound_violations'].append({
                'parameter': i,
                'value': value,
                'bounds': (lower, upper),
                'violation_amount': max(lower - value, value - upper)
            })
    
    # Additional validation checks
    if isinstance(solution, np.ndarray) and np.any(np.isnan(solution)):
        validation['is_valid'] = False
        validation['warnings'].append("Solution contains NaN values")
    
    if isinstance(solution, np.ndarray) and np.any(np.isinf(solution)):
        validation['is_valid'] = False  
        validation['warnings'].append("Solution contains infinite values")
    
    return validation

# Export utility functions
__all__ = [
    'save_results',
    'load_results', 
    'create_results_summary',
    'analyze_convergence',
    'compare_algorithms',
    'parameter_sensitivity_heatmap',
    'export_results_report',
    'calculate_performance_metrics',
    'create_parameter_table',
    'validate_solution'
]