# import pytest
# import sys
# from pathlib import Path
# import sympy as sp
# import numpy as np
#
# # 添加项目路径
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda_alg.QAOA.qaoa import QAOA, p_1
# from pyqpanda3.hamiltonian import PauliOperator
#
#
# class TestCalculateEnergy:
#     """测试calculate_energy接口"""
#
#     def test_basic_functionality_with_symbolic_problem(self):
#         """测试基本功能：使用符号问题计算能量"""
#         # 创建测试问题：f = 2*x0*x1 + 3*x2 - 1
#         vars = sp.symbols('x0:3')
#         f = 2*vars[0]*vars[1] + 3*vars[2] - 1
#
#         # 初始化QAOA
#         qaoa_f = QAOA(f)
#
#         # 测试不同的解
#         test_cases = [
#             ([1, 0, 0], 2*1*0 + 3*0 - 1),  # f(1,0,0) = -1
#             ([0, 1, 1], 2*0*1 + 3*1 - 1),  # f(0,1,1) = 2
#             ([1, 1, 0], 2*1*1 + 3*0 - 1),  # f(1,1,0) = 1
#             ([0, 0, 0], 2*0*0 + 3*0 - 1),  # f(0,0,0) = -1
#             ([1, 1, 1], 2*1*1 + 3*1 - 1),  # f(1,1,1) = 4
#         ]
#
#         for solution, expected_energy in test_cases:
#             calculated_energy = qaoa_f.calculate_energy(solution)
#             assert abs(calculated_energy - expected_energy) < 1e-10, \
#                 f"解 {solution} 的能量计算错误: 期望 {expected_energy}, 得到 {calculated_energy}"
#
# if __name__ == "__main__":
#     # 运行测试
#     pytest.main([__file__, "-v"])