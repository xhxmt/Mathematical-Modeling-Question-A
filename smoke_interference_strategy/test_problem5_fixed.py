#!/usr/bin/env python3
"""
问题5快速测试：验证修复后的算法能产生非零结果
"""

import sys
import os
import time
import numpy as np

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

def test_problem5_fixed():
    """测试修复后的问题5算法"""
    print("="*70)
    print("测试问题5修复版：多无人机多干扰弹对多导弹")
    print("="*70)
    
    try:
        from src.problem5_solver_fixed import Problem5SolverFixed
        
        print("创建求解器...")
        # Use very simplified parameters for testing
        solver = Problem5SolverFixed(
            num_drones=2,           # Reduced from 3
            grenades_per_drone=1,   # Reduced from 2
            num_missiles=2,         # Reduced from 3
            simulation_time_step=0.05  # Larger time step
        )
        
        print(f"配置: {solver.num_drones}架无人机 × {solver.grenades_per_drone}枚干扰弹 对 {solver.num_missiles}枚导弹")
        print(f"决策变量维度: {len(solver.bounds)}")
        
        start_time = time.time()
        
        print("\n开始优化（简化版本）...")
        results = solver.solve(max_iterations=15, population_size=8)
        
        execution_time = time.time() - start_time
        
        print(f"\n✅ 测试完成，用时: {execution_time:.2f}秒")
        
        # Check results
        obscuration = results['optimal_obscuration']
        print(f"\n📊 遮蔽效果:")
        for i in range(solver.num_missiles):
            duration = obscuration.get(f'missile_{i+1}', 0)
            print(f"  导弹 M{i+1}: {duration:.3f} 秒")
        
        total_duration = obscuration.get('total_obscuration', 0)
        print(f"  总遮蔽时长: {total_duration:.3f} 秒")
        print(f"  平均遮蔽时长: {obscuration.get('average_obscuration', 0):.3f} 秒")
        
        # Analyze results
        if total_duration > 0:
            print(f"\n🎉 算法修复成功！找到了有效的遮蔽策略")
            
            # Print strategy summary
            analysis = results['analysis']
            print(f"\n📈 策略摘要:")
            for drone in analysis['drone_deployments']:
                print(f"  无人机 FY{drone['drone_id']}: 方向{drone['flight_angle_degrees']:.0f}°, "
                      f"速度{drone['flight_speed']:.0f}m/s")
            
            timing = analysis['timing_analysis']
            if timing:
                print(f"\n⏰ 时序分析:")
                print(f"  首次起爆: {timing['earliest_detonation']:.1f}秒")
                print(f"  最后消散: {timing['latest_fade']:.1f}秒")
                print(f"  覆盖时长: {timing['total_coverage_span']:.1f}秒")
            
            return True
        else:
            print(f"\n⚠️  算法仍未找到有效解，可能需要进一步调整参数")
            return False
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("🧪 问题5修复版算法测试")
    print("="*50)
    
    try:
        success = test_problem5_fixed()
        
        if success:
            print(f"\n✅ 问题5算法修复验证成功！")
            print(f"主要改进:")
            print(f"  - 修复了导弹目标位置（指向真目标而非假目标）")
            print(f"  - 改进了参数范围和约束条件") 
            print(f"  - 简化了遮蔽计算逻辑")
            print(f"  - 添加了调试信息和错误处理")
        else:
            print(f"\n⚠️  算法仍需进一步优化")
        
        return success
        
    except KeyboardInterrupt:
        print("\n❌ 测试被用户中断")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)