# import sys
# from pathlib import Path
# import pytest
#
# # 添加项目路径到系统路径
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# # 导入测试所需的模块
# from pyqpanda_alg.QAOA import qaoa
# from pyqpanda3.hamiltonian import PauliOperator
#
#
# class TestP1Interface:
#     """测试 qaoa.p_1 接口的功能性"""
#
#     def test_p1_basic_functionality(self):
#         """测试 p_1 接口的基本功能"""
#         # 测试不同索引的 Pauli 算符生成
#         operator_0 = qaoa.p_1(0)
#         operator_1 = qaoa.p_1(1)
#         operator_2 = qaoa.p_1(2)
#
#         # 验证返回对象类型
#         assert isinstance(operator_0, PauliOperator)
#         assert isinstance(operator_1, PauliOperator)
#         assert isinstance(operator_2, PauliOperator)
#
#         # 验证不同索引生成的算符不同
#         assert str(operator_0) != str(operator_1)
#         assert str(operator_1) != str(operator_2)
#
# if __name__ == "__main__":
#     pytest.main([__file__, "-v"])