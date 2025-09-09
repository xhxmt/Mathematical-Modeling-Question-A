#!/usr/bin/env python3
"""
Main execution script for Smoke Interference Strategy Optimization

This script runs all five problems sequentially and generates comprehensive
results including optimization solutions, analysis, and visualizations.

Usage:
    python main.py [--problems 1,2,3,4,5] [--visualize] [--output-dir results/]
"""

import argparse
import sys
import os
import time
import warnings
from typing import List, Dict, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.problem1_solver import Problem1Solver
from src.problem2_solver import Problem2Solver
from src.problem3_solver import Problem3Solver
from src.problem4_solver import Problem4Solver
# Use the fixed version of Problem 5 solver
try:
    from src.problem5_solver_fixed import Problem5Solver
except ImportError:
    from src.problem5_solver import Problem5Solver

# Import visualization with fallback
try:
    from src.visualization import save_visualization_plots
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    def save_visualization_plots(*args, **kwargs):
        print("警告: 可视化功能不可用，跳过图表生成")

def run_problem1() -> Dict[str, Any]:
    """Run Problem 1: Fixed Strategy Obscuration Duration"""
    print("\n" + "="*80)
    print("开始求解问题1：固定策略下的遮蔽时长计算")
    print("="*80)
    
    start_time = time.time()
    
    # Create solver and run
    solver = Problem1Solver(simulation_time_step=0.01)
    results = solver.solve()
    
    # Print results
    solver.print_solution(results)
    
    execution_time = time.time() - start_time
    print(f"\n问题1求解完成，用时: {execution_time:.2f} 秒")
    
    return results

def run_problem2() -> Dict[str, Any]:
    """Run Problem 2: Single Grenade Optimal Strategy"""
    print("\n" + "="*80)
    print("开始求解问题2：单枚干扰弹的最优投放策略")
    print("="*80)
    
    start_time = time.time()
    
    # Create solver and run optimization
    solver = Problem2Solver(simulation_time_step=0.02)
    results = solver.solve_with_differential_evolution(max_iterations=80, population_size=12)
    
    # Print results
    solver.print_solution(results)
    
    execution_time = time.time() - start_time
    print(f"\n问题2求解完成，用时: {execution_time:.2f} 秒")
    
    return results

def run_problem3() -> Dict[str, Any]:
    """Run Problem 3: Single Drone Multiple Grenades"""
    print("\n" + "="*80)
    print("开始求解问题3：单无人机多干扰弹策略")
    print("="*80)
    
    start_time = time.time()
    
    # Create solver with 3 grenades
    solver = Problem3Solver(num_grenades=3, simulation_time_step=0.02)
    results = solver.solve(max_iterations=100, population_size=18)
    
    # Print results
    solver.print_solution(results)
    
    execution_time = time.time() - start_time
    print(f"\n问题3求解完成，用时: {execution_time:.2f} 秒")
    
    return results

def run_problem4() -> Dict[str, Any]:
    """Run Problem 4: Multiple Drones Single Grenade Each"""
    print("\n" + "="*80)
    print("开始求解问题4：多无人机单干扰弹策略")
    print("="*80)
    
    start_time = time.time()
    
    # Create solver with 3 drones
    solver = Problem4Solver(num_drones=3, simulation_time_step=0.02)
    results = solver.solve(max_iterations=80, population_size=20)
    
    # Print results
    solver.print_solution(results)
    
    execution_time = time.time() - start_time
    print(f"\n问题4求解完成，用时: {execution_time:.2f} 秒")
    
    return results

def run_problem5() -> Dict[str, Any]:
    """Run Problem 5: Multi-Drone Multi-Grenade vs Multi-Missile"""
    print("\n" + "="*80)
    print("开始求解问题5：多无人机多干扰弹对多导弹策略")
    print("="*80)
    
    start_time = time.time()
    
    # Create solver with reduced complexity for demonstration
    solver = Problem5Solver(num_drones=3, grenades_per_drone=2, num_missiles=3, 
                           simulation_time_step=0.03)
    results = solver.solve(max_iterations=50, population_size=20)
    
    # Print results
    solver.print_solution(results)
    
    execution_time = time.time() - start_time
    print(f"\n问题5求解完成，用时: {execution_time:.2f} 秒")
    
    return results

def save_results_summary(all_results: Dict[str, Any], output_dir: str):
    """Save comprehensive results summary"""
    os.makedirs(output_dir, exist_ok=True)
    
    summary_file = os.path.join(output_dir, "results_summary.txt")
    
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write("烟幕干扰弹投放策略优化 - 结果汇总\n")
        f.write("="*60 + "\n\n")
        
        for problem_name, results in all_results.items():
            f.write(f"{problem_name.upper()} 结果汇总:\n")
            f.write("-" * 40 + "\n")
            
            if problem_name == "problem1":
                f.write(f"有效遮蔽时长: {results['obscuration_duration']:.3f} 秒\n")
            elif problem_name in ["problem2"]:
                f.write(f"最大有效遮蔽时长: {results['optimal_duration']:.3f} 秒\n")
                params = results['optimal_parameters']
                f.write(f"最优飞行速度: {params['flight_speed']:.1f} m/s\n")
                f.write(f"最优投放时间: {params['drop_time']:.2f} s\n")
                f.write(f"最优起爆延迟: {params['detonation_delay']:.2f} s\n")
            elif problem_name in ["problem3", "problem4", "problem5"]:
                if 'optimal_duration' in results:
                    f.write(f"最大总遮蔽时长: {results['optimal_duration']:.3f} 秒\n")
                elif 'optimal_obscuration' in results:
                    total_duration = results['optimal_obscuration']['total_obscuration']
                    f.write(f"最大总遮蔽时长: {total_duration:.3f} 秒\n")
                else:
                    f.write(f"优化结果: 未找到有效解\n")
            
            f.write(f"优化评估次数: {results.get('optimization_stats', {}).get('total_evaluations', 'N/A')}\n")
            f.write("\n")
        
        f.write("\n生成时间: " + time.strftime("%Y-%m-%d %H:%M:%S") + "\n")
    
    print(f"结果汇总已保存到: {summary_file}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='运行烟幕干扰弹投放策略优化')
    
    # Support both positional and named arguments for problems
    parser.add_argument('problems_positional', nargs='*', 
                       help='问题编号 (例如: 1 2 3 或 1,2,3)')
    parser.add_argument('--problems', 
                       help='要运行的问题编号，逗号分隔 (例如: 1,2,3,4,5)')
    parser.add_argument('--visualize', action='store_true',
                       help='生成可视化图表')
    parser.add_argument('--output-dir', default='results/',
                       help='结果输出目录 (default: results/)')
    
    args = parser.parse_args()
    
    # Parse problem numbers - support both positional and named arguments
    if args.problems_positional:
        # Handle positional arguments: python3 main.py 1 2 3
        if len(args.problems_positional) == 1 and ',' in args.problems_positional[0]:
            # Handle: python3 main.py "1,2,3"
            problems_str = args.problems_positional[0]
        else:
            # Handle: python3 main.py 1 2 3
            problems_str = ','.join(args.problems_positional)
    elif args.problems:
        # Handle named argument: python3 main.py --problems 1,2,3
        problems_str = args.problems
    else:
        # Default to all problems
        problems_str = '1,2,3,4,5'
    
    try:
        problem_numbers = [int(x.strip()) for x in problems_str.split(',')]
    except ValueError:
        print("错误: 无效的问题编号格式")
        print("正确用法:")
        print("  python3 main.py 1")
        print("  python3 main.py 1 2 3")
        print("  python3 main.py --problems 1,2,3")
        sys.exit(1)
    
    # Validate problem numbers
    valid_problems = set(range(1, 6))
    if not set(problem_numbers).issubset(valid_problems):
        print("错误: 问题编号必须在1-5之间")
        sys.exit(1)
    
    print("烟幕干扰弹投放策略优化求解器")
    print("="*50)
    print(f"将要求解的问题: {problem_numbers}")
    print(f"可视化: {'是' if args.visualize and VISUALIZATION_AVAILABLE else '否' if args.visualize else '否'}")
    if args.visualize and not VISUALIZATION_AVAILABLE:
        print("注意: 可视化功能不可用，将跳过图表生成")
    print(f"输出目录: {args.output_dir}")
    
    # Suppress optimization warnings for cleaner output
    warnings.filterwarnings('ignore', category=UserWarning)
    
    # Dictionary to store all results
    all_results = {}
    total_start_time = time.time()
    
    # Problem runners mapping
    problem_runners = {
        1: run_problem1,
        2: run_problem2, 
        3: run_problem3,
        4: run_problem4,
        5: run_problem5
    }
    
    # Run selected problems
    for problem_num in sorted(problem_numbers):
        try:
            results = problem_runners[problem_num]()
            problem_key = f"problem{problem_num}"
            all_results[problem_key] = results
            
            # Generate visualization if requested
            if args.visualize:
                try:
                    save_visualization_plots(results, problem_key, args.output_dir)
                except Exception as e:
                    print(f"警告: 无法生成问题{problem_num}的可视化: {e}")
                    
        except KeyboardInterrupt:
            print(f"\n用户中断，停止执行问题{problem_num}")
            break
        except Exception as e:
            print(f"错误: 问题{problem_num}执行失败: {e}")
            continue
    
    # Calculate total execution time
    total_time = time.time() - total_start_time
    
    # Print final summary
    print("\n" + "="*80)
    print("全部问题求解完成!")
    print("="*80)
    print(f"总执行时间: {total_time:.2f} 秒")
    print(f"成功求解问题数量: {len(all_results)}")
    
    if all_results:
        print("\n结果汇总:")
        for problem_key, results in all_results.items():
            problem_num = problem_key[-1]
            if problem_key == "problem1":
                duration = results['obscuration_duration']
                print(f"  问题{problem_num}: 遮蔽时长 {duration:.3f} 秒")
            else:
                if 'optimal_duration' in results:
                    duration = results['optimal_duration']
                    print(f"  问题{problem_num}: 最优遮蔽时长 {duration:.3f} 秒")
                elif 'optimal_obscuration' in results:
                    total_duration = results['optimal_obscuration']['total_obscuration']
                    print(f"  问题{problem_num}: 总遮蔽时长 {total_duration:.3f} 秒")
        
        # Save comprehensive summary
        save_results_summary(all_results, args.output_dir)
        
    print(f"\n所有结果已保存到目录: {args.output_dir}")
    print("求解完成!")

if __name__ == "__main__":
    main()