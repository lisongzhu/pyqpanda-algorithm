# import sys
# from pathlib import Path
# import pytest
# import numpy as np
#
# # 添加项目路径到系统路径
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda3.core import QProg, CPUQVM
# from pyqpanda_alg.QAOA import default_circuits
#
#
# class TestInitDState:
#     """测试 init_d_state 接口"""
#
#     def setup_method(self):
#         """测试前的初始化"""
#         self.machine = CPUQVM()
#
#     def calculate_domain_hamming_weights(self, result_key, domains):
#         """计算测量结果中每个域的汉明权重"""
#         domain_weights = []
#         for domain in domains:
#             weight = sum(1 for idx in domain if result_key[idx] == '1')
#             domain_weights.append(weight)
#         return domain_weights
#
#     def test_init_d_state_integer_domains(self):
#         """测试整数分区方式的Dicke态初始化"""
#         n_qubits = 6
#         k = 2  # 每个域的汉明权重
#         domains = 2  # 分成2个域
#
#         prog = QProg(n_qubits)
#         qubits = prog.qubits()
#
#         # 创建Dicke态初始化电路
#         init_circuit_func = default_circuits.init_d_state(domains, k)
#         init_circuit = init_circuit_func(qubits)
#         prog << init_circuit
#
#         # 运行量子程序
#         self.machine.run(prog, shots=1000)
#         results = self.machine.result().get_prob_dict(qubits)
#
#         # 计算域的分区（均匀分成k份）
#         domain_size = n_qubits // domains
#         domain_list = [list(range(i * domain_size, (i + 1) * domain_size)) for i in range(domains)]
#
#         print(f"\nInteger domains ({domains}) Dicke state k={k} - Measurement results:")
#         valid_states = 0
#
#         for key, prob in results.items():
#             if prob > 0.001:  # 忽略概率很小的状态
#                 key_reversed = key[::-1]  # 反转键以匹配量子比特顺序
#                 domain_weights = self.calculate_domain_hamming_weights(key_reversed, domain_list)
#
#                 print(f"State {key_reversed}: domain weights = {domain_weights}, prob = {prob:.4f}")
#
#                 # 验证每个域的汉明权重都等于k
#                 for weight in domain_weights:
#                     assert weight == k, f"Domain weight should be {k}, but got {weight} for state {key_reversed}"
#
#                 valid_states += 1
#
#         assert valid_states > 0, "No valid Dicke states found"
#
#
# if __name__ == "__main__":
#     # 运行测试
#     pytest.main([__file__, "-v", "-s"])