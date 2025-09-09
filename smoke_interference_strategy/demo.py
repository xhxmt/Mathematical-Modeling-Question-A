#!/usr/bin/env python3
"""
演示脚本：运行问题1和问题2进行基础功能验证

这个脚本运行核心的问题1和问题2，用于快速验证系统功能正常运行。
运行时间较短，适合初始测试和演示。
"""

import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.problem1_solver import Problem1Solver
from src.problem2_solver import Problem2Solver

def demo_problem1():
    """演示问题1：固定策略分析"""
    print("="*60)
    print("演示问题1：固定策略下的遮蔽时长计算")
    print("="*60)
    
    # 使用较大的时间步长以加快演示速度
    solver = Problem1Solver(simulation_time_step=0.02)
    
    print("正在计算固定策略下的遮蔽效果...")
    start_time = time.time()
    
    results = solver.solve()
    
    execution_time = time.time() - start_time
    print(f"\n计算完成，用时: {execution_time:.2f} 秒")
    
    # 输出关键结果
    print(f"\n【核心结果】")
    print(f"有效遮蔽时长: {results['obscuration_duration']:.3f} 秒")
    
    # 分析数据
    analysis = results['distance_analysis']
    print(f"\n【几何分析】")
    print(f"烟幕云团起爆高度: {analysis['cloud_altitude']:.1f} 米")
    print(f"烟幕到真目标距离: {analysis['cloud_to_target']:.1f} 米")
    
    return results

def demo_problem2():
    """演示问题2：策略优化"""
    print("\n" + "="*60)
    print("演示问题2：单枚干扰弹的最优投放策略")
    print("="*60)
    
    solver = Problem2Solver(simulation_time_step=0.03)
    
    print("正在进行策略优化...")
    print("使用差分进化算法，这可能需要1-2分钟...")
    
    start_time = time.time()
    
    # 使用较少的迭代次数用于快速演示
    results = solver.solve_with_differential_evolution(
        max_iterations=40, 
        population_size=10
    )
    
    execution_time = time.time() - start_time
    print(f"\n优化完成，用时: {execution_time:.2f} 秒")
    
    # 输出关键结果
    print(f"\n【优化结果】")
    print(f"最大遮蔽时长: {results['optimal_duration']:.3f} 秒")
    
    params = results['optimal_parameters']
    analysis = results['analysis']
    print(f"\n【最优策略】")
    print(f"飞行方向: {analysis['flight_direction_degrees']:.1f}°")
    print(f"飞行速度: {params['flight_speed']:.1f} m/s")
    print(f"投放时间: {params['drop_time']:.2f} 秒")
    print(f"起爆延迟: {params['detonation_delay']:.2f} 秒")
    
    # 性能提升分析
    return results

def main():
    """主演示函数"""
    print("烟幕干扰弹投放策略优化 - 功能演示")
    print("="*60)
    print("本演示将运行问题1和问题2，展示系统核心功能")
    print("预计总耗时: 1-3分钟\n")
    
    total_start = time.time()
    
    # 运行问题1
    problem1_results = demo_problem1()
    
    # 运行问题2
    problem2_results = demo_problem2()
    
    # 总结比较
    print("\n" + "="*60)
    print("演示总结与对比分析")
    print("="*60)
    
    fixed_duration = problem1_results['obscuration_duration']
    optimal_duration = problem2_results['optimal_duration']
    improvement = optimal_duration - fixed_duration
    improvement_pct = (improvement / fixed_duration * 100) if fixed_duration > 0 else 0
    
    print(f"固定策略遮蔽时长: {fixed_duration:.3f} 秒")
    print(f"优化策略遮蔽时长: {optimal_duration:.3f} 秒")
    print(f"性能提升: {improvement:.3f} 秒 ({improvement_pct:+.1f}%)")
    
    if improvement > 0:
        print(f"\n✓ 优化策略相比固定策略有显著提升!")
    else:
        print(f"\n⚠ 优化结果可能需要更多迭代次数或参数调整")
    
    total_time = time.time() - total_start
    print(f"\n演示总用时: {total_time:.2f} 秒")
    
    print("\n演示完成! 系统功能验证正常。")
    print("如需运行完整的5个问题，请使用: python main.py")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n演示被用户中断")
    except Exception as e:
        print(f"\n演示过程中出现错误: {e}")
        sys.exit(1)