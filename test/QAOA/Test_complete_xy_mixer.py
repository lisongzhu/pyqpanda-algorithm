# import sys
# from pathlib import Path
# import pytest
# import numpy as np
#
# # 添加项目路径到系统路径
# sys.path.append((Path.cwd().parent.parent).__str__())
#
# from pyqpanda3.core import QProg, RX, CPUQVM
# from pyqpanda_alg.QAOA import default_circuits
#
#
# class TestCompleteXYMixer:
#     """测试 complete_xy_mixer 接口"""
#
#     def setup_method(self):
#         """测试前的初始化"""
#         self.machine = CPUQVM()
#
#     def calculate_hamming_weight_distribution(self, prob_dict, max_weight):
#         """计算汉明权重分布"""
#         weight_probs = {}
#         for weight in range(max_weight + 1):
#             prob = sum(value for key, value in prob_dict.items()
#                       if key.count('1') == weight)
#             weight_probs[weight] = prob
#         return weight_probs
#
#     def test_complete_xy_mixer_hamming_weight_preservation(self):
#         """测试XY mixer应该保持汉明权重"""
#         n_qubits = 4
#         prog = QProg(n_qubits)
#         qubits = prog.qubits()
#
#         # 创建初始状态 - 使用随机RX门创建复杂初始状态
#         for q in qubits:
#             prog << RX(q, np.random.random() * 2 * np.pi)
#
#         # 获取初始状态的概率分布
#         self.machine.run(prog, shots=1)
#         original_result = self.machine.result().get_prob_dict()
#
#         # 计算初始汉明权重分布
#         original_weight_probs = self.calculate_hamming_weight_distribution(original_result, n_qubits)
#
#         # 应用 complete_xy_mixer
#         beta = np.pi / 5
#         circuit = default_circuits.complete_xy_mixer(qubits, beta)
#         prog << circuit
#
#         # 获取应用mixer后的概率分布
#         self.machine.run(prog, shots=1)
#         final_result = self.machine.result().get_prob_dict()
#
#         # 计算最终汉明权重分布
#         final_weight_probs = self.calculate_hamming_weight_distribution(final_result, n_qubits)
#
#         # 打印详细的权重分布变化
#         print("\nHamming weight distribution comparison:")
#         for weight in range(n_qubits + 1):
#             original_prob = original_weight_probs.get(weight, 0)
#             final_prob = final_weight_probs.get(weight, 0)
#             print(f"Hamming weight {weight}: {original_prob:.4f} -> {final_prob:.4f}")
#
#             # XY mixer应该保持汉明权重（在统计误差范围内）
#             # 由于有限shots和数值精度，允许一定的误差
#             tolerance = 0.05  # 5% 容忍度
#             assert abs(original_prob - final_prob) < tolerance, \
#                 f"Hamming weight {weight} not preserved within tolerance: {original_prob:.4f} -> {final_prob:.4f}"
#
#
# if __name__ == "__main__":
#     # 运行测试
#     pytest.main([__file__, "-v", "-s"])