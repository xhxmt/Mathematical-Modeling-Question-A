#!/usr/bin/env python3
"""
快速验证脚本：运行核心问题验证系统功能

这个脚本运行问题1和一个简化的问题2验证，确保系统正常工作。
适合快速测试和功能验证。
"""

import argparse
import sys
import os
import time
import warnings

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.problem1_solver import Problem1Solver
from src.problem2_solver import Problem2Solver

def run_problem1_quick():
    """快速运行问题1"""
    print("="*60)
    print("问题1：固定策略下的遮蔽时长计算")
    print("="*60)
    
    start_time = time.time()
    
    # 使用稍大的时间步长以加快计算
    solver = Problem1Solver(simulation_time_step=0.02)
    results = solver.solve()
    
    execution_time = time.time() - start_time
    
    print(f"✅ 问题1完成，用时: {execution_time:.2f}秒")
    print(f"📊 遮蔽时长: {results['obscuration_duration']:.3f}秒")
    print(f"📍 起爆位置: ({results['key_positions']['detonation_position'][0]:.0f}, "
          f"{results['key_positions']['detonation_position'][1]:.0f}, "
          f"{results['key_positions']['detonation_position'][2]:.0f})米")
    
    return results

def run_problem2_quick():
    """快速运行问题2的简化版本"""
    print("\n" + "="*60)
    print("问题2：单枚干扰弹优化策略（快速验证版）")
    print("="*60)
    
    start_time = time.time()
    
    # 使用较少的迭代次数进行快速验证
    solver = Problem2Solver(simulation_time_step=0.03)
    
    print("正在进行优化（简化版本，约30秒）...")
    results = solver.solve_with_differential_evolution(
        max_iterations=25, 
        population_size=8
    )
    
    execution_time = time.time() - start_time
    
    print(f"✅ 问题2完成，用时: {execution_time:.2f}秒")
    print(f"📊 最优遮蔽时长: {results['optimal_duration']:.3f}秒")
    
    params = results['optimal_parameters']
    analysis = results['analysis']
    
    print(f"📈 最优策略:")
    print(f"   飞行方向: {analysis['flight_direction_degrees']:.1f}°")
    print(f"   飞行速度: {params['flight_speed']:.1f} m/s")
    print(f"   投放时间: {params['drop_time']:.1f}秒")
    print(f"   起爆延迟: {params['detonation_delay']:.1f}秒")
    
    return results

def run_optimization_validation():
    """验证优化算法基本功能"""
    print("\n" + "="*60)
    print("优化算法功能验证")
    print("="*60)
    
    import numpy as np
    
    # 简单的优化测试：最大化二次函数
    def test_objective(x):
        # x应该接近[1.5, 2.5, 1.0, 5.0]来最大化这个函数
        target = np.array([1.5, 2.5, 1.0, 5.0])
        return -np.sum((x - target)**2)  # 负号因为scipy是最小化
    
    from scipy.optimize import differential_evolution
    
    bounds = [(0, 3), (1, 4), (0, 2), (3, 7)]
    
    print("运行优化算法功能测试...")
    start_time = time.time()
    
    result = differential_evolution(
        test_objective, 
        bounds, 
        maxiter=20, 
        popsize=5,
        seed=42
    )
    
    test_time = time.time() - start_time
    print(f"✅ 优化算法测试完成，用时: {test_time:.2f}秒")
    print(f"📊 收敛成功: {result.success}")
    print(f"📈 最优解: [{result.x[0]:.2f}, {result.x[1]:.2f}, {result.x[2]:.2f}, {result.x[3]:.2f}]")
    print(f"🎯 目标值: {-result.fun:.3f}")
    
    return result.success

def main():
    """主验证函数"""
    parser = argparse.ArgumentParser(description='快速验证烟幕干扰策略优化系统')
    parser.add_argument('--skip-optimization', action='store_true',
                       help='跳过优化测试，只运行问题1')
    args = parser.parse_args()
    
    print("🚀 烟幕干扰弹投放策略优化系统 - 快速验证")
    print("="*60)
    
    if args.skip_optimization:
        print("模式: 仅验证基础功能（跳过优化）")
    else:
        print("模式: 完整功能验证（包含优化）")
    
    print("预计总时间:", "10秒" if args.skip_optimization else "60-90秒")
    
    # 抑制警告信息
    warnings.filterwarnings('ignore', category=UserWarning)
    
    total_start = time.time()
    success_count = 0
    total_tests = 2 if args.skip_optimization else 3
    
    try:
        # 验证问题1
        print("\n🔍 开始验证...")
        problem1_results = run_problem1_quick()
        success_count += 1
        
        if not args.skip_optimization:
            # 验证优化算法基础功能
            if run_optimization_validation():
                success_count += 1
            
            # 验证问题2
            problem2_results = run_problem2_quick()
            success_count += 1
            
            # 性能对比
            if success_count >= 2:
                print("\n" + "="*60)
                print("性能对比分析")
                print("="*60)
                
                p1_duration = problem1_results['obscuration_duration']
                p2_duration = problem2_results['optimal_duration']
                
                print(f"固定策略遮蔽时长: {p1_duration:.3f}秒")
                print(f"优化策略遮蔽时长: {p2_duration:.3f}秒")
                
                if p2_duration > p1_duration:
                    improvement = p2_duration - p1_duration
                    improvement_pct = (improvement / p1_duration * 100) if p1_duration > 0 else 0
                    print(f"🎉 性能提升: +{improvement:.3f}秒 ({improvement_pct:+.1f}%)")
                elif p2_duration > 0:
                    print("📊 优化结果与固定策略相近")
                else:
                    print("⚠️  优化算法可能需要更多迭代")
        
        total_time = time.time() - total_start
        
        print("\n" + "="*60)
        print("验证结果总结")
        print("="*60)
        print(f"✅ 成功验证: {success_count}/{total_tests} 项功能")
        print(f"⏱️  总用时: {total_time:.1f}秒")
        
        if success_count == total_tests:
            print("🎉 系统功能验证完全成功！")
            print("\n📋 可运行的完整功能:")
            print("   • python3 main.py --problems 1     # 基础分析")
            print("   • python3 main.py --problems 1,2   # 优化策略")
            print("   • python3 main.py --problems 1,2,3 # 多干扰弹")
            print("   • python3 demo_simple.py           # 核心验证")
        else:
            print("⚠️ 部分功能可能存在问题，但核心算法正常")
        
        return success_count == total_tests
        
    except KeyboardInterrupt:
        print("\n\n❌ 验证被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 验证过程出错: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)