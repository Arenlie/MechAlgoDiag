# model/__init__.py

# 版本号
__version__ = "0.1.0"

# 从子模块导入到包级别
from .feature_calculate import acc2dis
from .unbalance import detect_unbalance_fault as detect_unbalance_fault
from .misalignment import detect_misalignment_fault as detect_misalignment_fault
from .coupling_wear import detect_coupling_fault as detect_coupling_fault
from .bearing_fault_lr import detect_bearing_fault as detect_bearing_fault
from .looseness import detect_looseness_fault as detect_looseness_fault
from .rpm_detection import Speed_Estimate_algorithm as Speed_Estimate_algorithm


# 控制 `from model import *` 的可见符号
__all__ = [
    "acc2dis",
    "detect_unbalance_fault",
    "detect_misalignment_fault",
    "detect_coupling_fault",
    "detect_bearing_fault",
    "detect_looseness_fault",
    "Speed_Estimate_algorithm",
]
