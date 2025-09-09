"""
Test suite for smoke interference strategy optimization.

This module contains unit tests for the core physics calculations,
optimization algorithms, and visualization components.
"""

import unittest
import numpy as np
import sys
import os

# Add src directory to path for testing
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.physics_core import (
    PhysicsConstants, Vector3D, TrajectoryCalculator, 
    ObscurationAnalyzer, SimulationEngine
)
from src.problem1_solver import Problem1Solver

class TestPhysicsCore(unittest.TestCase):
    """Test core physics calculations"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_position1 = np.array([0.0, 0.0, 0.0])
        self.test_position2 = np.array([3.0, 4.0, 0.0])
        self.test_velocity = np.array([10.0, 0.0, 0.0])
        
    def test_vector3d_distance(self):
        """Test 3D distance calculation"""
        distance = Vector3D.distance(self.test_position1, self.test_position2)
        expected_distance = 5.0  # 3-4-5 triangle
        self.assertAlmostEqual(distance, expected_distance, places=6)
        
    def test_vector3d_normalize(self):
        """Test vector normalization"""
        test_vector = np.array([3.0, 4.0, 0.0])
        normalized = Vector3D.normalize(test_vector)
        expected_norm = np.array([0.6, 0.8, 0.0])
        np.testing.assert_array_almost_equal(normalized, expected_norm)
        
    def test_missile_trajectory(self):
        """Test missile trajectory calculation"""
        initial_pos = np.array([1000.0, 0.0, 100.0])
        target_pos = np.array([0.0, 0.0, 0.0])
        speed = 300.0
        time = 1.0
        
        result_pos = TrajectoryCalculator.missile_position(
            initial_pos, target_pos, speed, time)
        
        # Check that missile moves towards target
        self.assertLess(result_pos[0], initial_pos[0])  # X decreases
        self.assertLess(result_pos[2], initial_pos[2])  # Z decreases
        
    def test_drone_trajectory(self):
        """Test drone trajectory calculation"""
        initial_pos = np.array([100.0, 0.0, 50.0])
        velocity = np.array([10.0, 5.0, 0.0])
        time = 2.0
        
        result_pos = TrajectoryCalculator.drone_position(initial_pos, velocity, time)
        expected_pos = np.array([120.0, 10.0, 50.0])
        
        np.testing.assert_array_almost_equal(result_pos, expected_pos)
        
    def test_grenade_trajectory(self):
        """Test grenade projectile motion"""
        drop_pos = np.array([100.0, 0.0, 50.0])
        initial_velocity = np.array([10.0, 0.0, 0.0])
        time = 1.0
        
        result_pos = TrajectoryCalculator.grenade_trajectory(
            drop_pos, initial_velocity, time)
        
        # Check that grenade falls due to gravity
        self.assertLess(result_pos[2], drop_pos[2])
        # Check horizontal movement
        self.assertGreater(result_pos[0], drop_pos[0])
        
    def test_smoke_cloud_motion(self):
        """Test smoke cloud sinking motion"""
        detonation_pos = np.array([100.0, 0.0, 50.0])
        time = 2.0
        
        result_pos = TrajectoryCalculator.smoke_cloud_center(detonation_pos, time)
        
        # Cloud should sink downward
        expected_z = detonation_pos[2] + PhysicsConstants.CLOUD_SINK_VELOCITY[2] * time
        self.assertAlmostEqual(result_pos[2], expected_z)
        
    def test_line_sphere_intersection(self):
        """Test line-sphere intersection calculation"""
        line_start = np.array([0.0, 0.0, 0.0])
        line_end = np.array([10.0, 0.0, 0.0])
        sphere_center = np.array([5.0, 0.0, 0.0])
        sphere_radius = 2.0
        
        # Line passes through sphere
        result = ObscurationAnalyzer.check_line_sphere_intersection(
            line_start, line_end, sphere_center, sphere_radius)
        self.assertTrue(result)
        
        # Line misses sphere
        sphere_center_miss = np.array([5.0, 5.0, 0.0])
        result_miss = ObscurationAnalyzer.check_line_sphere_intersection(
            line_start, line_end, sphere_center_miss, sphere_radius)
        self.assertFalse(result_miss)

class TestProblem1Solver(unittest.TestCase):
    """Test Problem 1 solver functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.solver = Problem1Solver(simulation_time_step=0.05)  # Larger step for faster testing
        
    def test_solver_initialization(self):
        """Test solver proper initialization"""
        self.assertIsNotNone(self.solver.missile_initial_pos)
        self.assertIsNotNone(self.solver.drone_initial_pos)
        self.assertGreater(len(self.solver.target_key_points), 0)
        
    def test_target_key_points_generation(self):
        """Test target key points generation"""
        points = self.solver.target_key_points
        self.assertGreater(len(points), 0)
        
        # Check that all points are 3D arrays
        for point in points:
            self.assertEqual(len(point), 3)
            
    def test_drone_velocity_calculation(self):
        """Test drone velocity calculation"""
        velocity = self.solver._calculate_drone_velocity()
        
        # Check that velocity has correct magnitude
        speed_magnitude = np.linalg.norm(velocity)
        self.assertAlmostEqual(speed_magnitude, self.solver.drone_speed, places=1)
        
        # Check that z-component is zero (constant altitude)
        self.assertAlmostEqual(velocity[2], 0.0, places=6)
        
    def test_solve_method(self):
        """Test that solve method runs without errors"""
        try:
            results = self.solver.solve()
            
            # Check that results contain expected keys
            self.assertIn('obscuration_duration', results)
            self.assertIn('strategy_parameters', results)
            self.assertIn('key_positions', results)
            
            # Check that obscuration duration is non-negative
            self.assertGreaterEqual(results['obscuration_duration'], 0.0)
            
        except Exception as e:
            self.fail(f"Solve method raised an exception: {e}")

class TestSimulationEngine(unittest.TestCase):
    """Test simulation engine functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.engine = SimulationEngine(time_step=0.1)
        
    def test_simulation_engine_initialization(self):
        """Test simulation engine initialization"""
        self.assertEqual(self.engine.time_step, 0.1)
        self.assertEqual(self.engine.current_time, 0.0)
        
    def test_reset_functionality(self):
        """Test simulation reset"""
        self.engine.current_time = 10.0
        self.engine.reset()
        self.assertEqual(self.engine.current_time, 0.0)

class TestOptimizationComponents(unittest.TestCase):
    """Test optimization-related components"""
    
    def test_bounds_validation(self):
        """Test that optimization bounds are reasonable"""
        # These would be imported from problem solvers
        min_speed = PhysicsConstants.DRONE_SPEED_MIN
        max_speed = PhysicsConstants.DRONE_SPEED_MAX
        
        self.assertGreater(max_speed, min_speed)
        self.assertGreater(min_speed, 0)
        
    def test_physics_constants(self):
        """Test physics constants are reasonable"""
        self.assertGreater(PhysicsConstants.MISSILE_SPEED, 0)
        self.assertGreater(PhysicsConstants.CLOUD_RADIUS, 0)
        self.assertGreater(PhysicsConstants.CLOUD_LIFETIME, 0)
        self.assertLess(PhysicsConstants.CLOUD_SINK_VELOCITY[2], 0)  # Downward

def run_tests():
    """Run all unit tests"""
    print("运行烟幕干扰策略优化测试套件...")
    
    # Create test loader
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestPhysicsCore,
        TestProblem1Solver,
        TestSimulationEngine,
        TestOptimizationComponents
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestClass(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print results summary
    print(f"\n测试结果汇总:")
    print(f"运行测试: {result.testsRun}")
    print(f"失败: {len(result.failures)}")
    print(f"错误: {len(result.errors)}")
    print(f"成功率: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    return result.wasSuccessful()

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)