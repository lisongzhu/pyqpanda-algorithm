# import pytest
# import sys
# from pathlib import Path
# import sympy as sp
# import numpy as np
#
# # 添加项目路径
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda_alg.QAOA import qaoa
# from pyqpanda3.core import CPUQVM, QProg, H
#
#
# class TestPauliZOperatorToCircuit:
#     """测试pauli_z_operator_to_circuit接口"""
#
#     @classmethod
#     def setup_class(cls):
#         """测试类初始化"""
#         cls.machine = CPUQVM()
#
#     def setup_method(self):
#         """每个测试方法前的设置"""
#         self.prog = QProg(3)
#         self.qubits = self.prog.qubits()
#
#     def test_basic_functionality(self):
#         """测试基本功能：构造exp(-iH γ)的量子线路"""
#         # 创建测试问题：f = 2*x0*x1 + 3*x2 - 1
#         vars = sp.symbols('x0:3')
#         f = 2*vars[0]*vars[1] + 3*vars[2] - 1
#
#         # 将问题转换为Pauli Z算子
#         operator = qaoa.problem_to_z_operator(f)
#
#         # 构造量子线路
#         gamma = 1.0
#         circuit, _ = qaoa.pauli_z_operator_to_circuit(operator, self.qubits, gamma)
#
#         # 验证线路不为空
#         assert circuit is not None
#         originir_str = circuit.originir()
#         assert isinstance(originir_str, str)
#         assert len(originir_str) > 0
#
#
# if __name__ == "__main__":
#     # 运行测试
#     pytest.main([__file__, "-v", "-s"])