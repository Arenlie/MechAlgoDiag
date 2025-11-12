# -*- coding: utf-8 -*-
"""
机理诊断计算与汇总
"""

import numpy as np

from model import acc2dis, detect_unbalance_fault, detect_misalignment_fault, detect_coupling_fault, \
    detect_bearing_fault, detect_looseness_fault


def model_diagnosis(acc: np.ndarray, fs: float, fr: float, notice_th: float, warn_th: float):
    """
    诊断振动信号
    :param acc: np.ndarray, 振动信号
    :param fs: float, 采样频率
    :param fr: float, 转频
    :param notice_th: float, 预警阈值
    :param warn_th: float, 报警阈值
    :return: dict, 机理诊断结果
    """
    # 计算速度和位移
    s, d = acc2dis(acc, fs=fs)
    result = []
    # 检测机理故障
    res_looseness = detect_looseness_fault(s, acc, fs=fs, fr=fr, notice_th=notice_th, warn_th=warn_th)
    res_misalignment = detect_misalignment_fault(s, acc, fs=fs, fr=fr, notice_th=notice_th, warn_th=warn_th)
    res_unbalance = detect_unbalance_fault(s, acc, fs=fs, fr=fr, notice_th=notice_th, warn_th=warn_th)
    res_coupling = detect_coupling_fault(s, acc, fs=fs, fr=fr, notice_th=notice_th, warn_th=warn_th)
    res_bearing_kur = detect_bearing_fault(acc, fs=fs, notice_th=notice_th, warn_th=warn_th)
    res_bearing_lr = detect_bearing_fault(acc, fs=fs, notice_th=notice_th, warn_th=warn_th)

    if res_misalignment["alarm_level"] is not None:
        result.append(res_misalignment)
    if res_unbalance["alarm_level"] is not None:
        result.append(res_unbalance)
    if res_looseness["alarm_level"] is not None:
        result.append(res_looseness)
    if res_coupling["alarm_level"] is not None:
        result.append(res_coupling)
    if res_bearing_kur["alarm_level"] is not None:
        result.append(res_bearing_kur)
    if res_bearing_lr["alarm_level"] is not None:
        result.append(res_bearing_lr)

    return result
