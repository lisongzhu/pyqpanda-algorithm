# import sys
# from pathlib import Path
# import pytest
# import numpy as np
#
# # 添加项目路径到系统路径
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda3.core import QProg, QCircuit, RX, CPUQVM
# from pyqpanda_alg.QAOA import default_circuits
#
#
# class TestXYMixer:
#     """测试 xy_mixer 接口"""
#
#     def setup_method(self):
#         """测试前的初始化"""
#         self.machine = CPUQVM()
#
#     def calculate_domain_hamming_weights(self, prob_dict, domains):
#         """计算每个域的汉明权重分布"""
#         domain_probs = {}
#         for result_key, prob_value in prob_dict.items():
#             domain_weights = []
#             for domain in domains:
#                 # 计算该域的汉明权重
#                 weight = sum(1 for idx in domain if result_key[idx] == '1')
#                 domain_weights.append(weight)
#
#             # 将权重元组作为键
#             weight_key = tuple(domain_weights)
#             domain_probs[weight_key] = domain_probs.get(weight_key, 0) + prob_value
#
#         return domain_probs
#
#     def test_xy_mixer_integer_domains_parity(self):
#         """测试整数分区方式的parity XY mixer"""
#         n_qubits = 4
#         k_domains = 2  # 分成2个域
#
#         prog = QProg(n_qubits)
#         qubits = prog.qubits()
#
#         # 创建初始状态
#         for q in qubits:
#             prog << RX(q, np.random.random() * np.pi)
#
#         # 获取初始状态的概率分布
#         self.machine.run(prog, shots=1)
#         origin_result = self.machine.result().get_prob_dict()
#
#         # 创建XY mixer电路
#         circuit = default_circuits.xy_mixer(k_domains, 'PXY')(qubits, np.pi/2)
#         prog << circuit
#
#         # 获取最终状态的概率分布
#         self.machine.run(prog, shots=1)
#         final_result = self.machine.result().get_prob_dict()
#
#         # 计算域的分区（均匀分成k份）
#         domain_size = n_qubits // k_domains
#         domains = [list(range(i * domain_size, (i + 1) * domain_size)) for i in range(k_domains)]
#
#         # 计算每个域的汉明权重分布
#         origin_domain_probs = self.calculate_domain_hamming_weights(origin_result, domains)
#         final_domain_probs = self.calculate_domain_hamming_weights(final_result, domains)
#
#         print(f"\nInteger domains ({k_domains}) Parity XY Mixer - Domain Hamming Weight comparison:")
#         for weight_combo in origin_domain_probs:
#             origin_prob = origin_domain_probs[weight_combo]
#             final_prob = final_domain_probs.get(weight_combo, 0)
#             print(f"Domain weights {weight_combo}: {origin_prob:.4f} -> {final_prob:.4f}")
#
#             # 每个域的汉明权重应该保持
#             tolerance = 0.06
#             assert abs(origin_prob - final_prob) < tolerance, \
#                 f"Domain weights {weight_combo} not preserved: {origin_prob:.4f} -> {final_prob:.4f}"
#
# if __name__ == "__main__":
#     # 运行测试
#     pytest.main([__file__, "-v", "-s"])