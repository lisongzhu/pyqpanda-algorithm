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
# class TestP0Interface:
#     """测试 qaoa.p_0 接口的功能性"""
#
#     def test_p0_basic_functionality(self):
#         """测试 p_0 接口的基本功能 - 返回 PauliOperator 对象"""
#         # 测试不同索引的 Pauli 算符生成
#         operator_0 = qaoa.p_0(0)
#
#         # 验证返回对象类型
#         assert isinstance(operator_0, PauliOperator)
#
#         # 验证算符有字符串表示
#         assert str(operator_0) is not None
#         assert len(str(operator_0)) > 0
#
#         print(f"p_0(0) = {operator_0}")
#
#
# if __name__ == "__main__":
#     # 运行测试
#     pytest.main([__file__, "-v", "-s"])