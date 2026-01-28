# -*-coding:utf-8-*-
import sys
from pathlib import Path
import pytest
import numpy as np

# 添加项目路径到系统路径
sys.path.append((Path.cwd().parent.parent).__str__())


class TestQPCA:
    """QPCA测试类"""

    @pytest.fixture
    def sample_data(self):
        """提供标准测试数据"""
        return np.array([[-1, 2], [-2, -1], [-1, -2], [1, 3], [2, 1], [3, 2]])

    def test_qpca_normal(self, sample_data):
        """测试QPCA正常功能"""
        from pyqpanda_alg.QPCA import qpca

        # 执行QPCA
        data_q = qpca(sample_data, 1)

        assert data_q.shape == (6,1)
        assert isinstance(data_q, np.ndarray)



