import sys
from pathlib import Path
import pytest
import numpy as np

# 添加项目路径到系统路径
sys.path.append((Path.cwd().parent.parent).__str__())

# 导入测试所需的模块
from pyqpanda_alg.QAOA import qaoa


class TestParameterInterpolate:
    """测试 qaoa.parameter_interpolate 接口的功能性"""
    
    def test_basic_functionality(self):
        """测试基本功能 - 2p个参数插值为2(p+1)个参数"""
        # 使用示例中的测试用例
        initial_parameter = np.array([0.1, 0.2, 0.2, 0.1])
        new_parameter = qaoa.parameter_interpolate(initial_parameter)
        
        # 验证返回对象类型和形状
        assert isinstance(new_parameter, np.ndarray)
        assert new_parameter.shape == (6,)  # 2*(2+1)=6
        
        print(f"输入: {initial_parameter}")
        print(f"输出: {new_parameter}")


if __name__ == "__main__":
    # 运行测试
    pytest.main([__file__, "-v", "-s"])