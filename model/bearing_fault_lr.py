import numpy as np
from math import floor
from scipy.signal import argrelextrema
from model.feature_calculate import fft_filter, acc2dis, fft_spectrum, calc_acc_p, calc_acc_rms


# 通频速度有效值（10-1000Hz）
def calc_vel_pass_rms(signal, fs):
    """
    signal: 振动加速度信号
    fs: 采样频率
    :return: 通频速度有效值（10-1000Hz）
    """
    x1 = fft_filter(signal, 5, 1000, fs)
    x2, x3 = acc2dis(x1, fs)
    x4 = fft_filter(x2, 5, 1000, fs)
    vel_pass_rms = np.sqrt(np.mean(x4 ** 2))
    return vel_pass_rms


def Calc_kurtosis_with_optimization(data, fs):
    """
    data: 冲击脉冲信号
    fs: 采样频率
    Return
    acc_kurt: 峭度指标
    """
    # 1. 带通滤波
    data_filter = fft_filter(data, 3, 10000, fs)

    # 2. 箱型图滤波去除异常值
    q1 = np.percentile(data_filter, 5)
    q3 = np.percentile(data_filter, 95)
    # print(q1, q3)
    iqr = q3 - q1
    lower_bound = q1 - 1.0 * iqr
    upper_bound = q3 + 1.0 * iqr
    # print(lower_bound, upper_bound)
    data_filtered = np.array([x if lower_bound <= x <= upper_bound else 0 for x in data])

    if len(data_filtered) < 10:  # 防止样本太少
        return 0

    # 3. 峭度计算
    # m = np.mean(data_filtered)
    # std = np.std(data_filtered)
    # acc_kurt = np.mean((data_filtered - m) ** 4) / (std ** 4)
    K = sum(data_filtered ** 4) / len(data_filtered)
    a = np.sqrt(sum(data_filtered ** 2) / len(data_filtered))
    acc_kurt = K / (a ** 4)

    return acc_kurt


def Calc_SPM_impulse_rank(data, fs, top_n=40, window_sec=1):
    """
    通用函数：计算排名第 top_n 的局部极值，作为冲击指标。

    :param data: 1D 信号数据
    :param fs: 采样频率（Hz）
    :param top_n: 取第 top_n 个局部极值（例如：HR=1000，LR=40）
    :param window_sec: 每段时长（秒）
    :return: 冲击指标均值（float）
    """
    window_size = int(fs * window_sec)
    # print(window_size, len(data))
    num_segments = floor(len(data) / window_size)
    values = []

    for i in range(num_segments):
        segment = data[i * window_size: (i + 1) * window_size]
        peaks = segment[argrelextrema(segment, np.greater)]
        sorted_peaks = sorted(np.abs(peaks), reverse=True)
        # print(len(sorted_peaks))
        if len(sorted_peaks) >= top_n:
            value = sorted_peaks[top_n - 1]
        elif len(sorted_peaks) > 0:
            value = sorted_peaks[-1]
        else:
            value = 0
        values.append(value)

    return np.mean(values) if values else 0.0


# 定义函数来计算特征值（例如，计算均值和标准差等）
def detect_bearing_fault(acc_signal, fs, notice_th=8, warn_th=16):
    acc_signal = np.asarray(acc_signal, dtype=float)
    # 计算特征值
    acc_rms = calc_acc_rms(acc_signal, fs)
    # acc_kurtosis = Calc_kurtosis_with_optimization(acc_signal, fs)
    spm_lr = Calc_SPM_impulse_rank(acc_signal, fs, top_n=40, window_sec=1)
    peak_amp = calc_acc_p(acc_signal, fs)

    # 加速度峰值因子
    acc_p_factor = peak_amp / acc_rms

    if (notice_th <= spm_lr < warn_th) and acc_p_factor >= 4:
        alarm_level = "注意"
        alarm_threshold = notice_th
    elif (spm_lr >= warn_th) and acc_p_factor >= 4:
        alarm_level = "警告"
        alarm_threshold = warn_th
    else:
        alarm_level = None
        alarm_threshold = None
    alarm_message = f"轴承磨损低冲能量【{alarm_level}级】报警，低冲能量为{round(spm_lr, 2)}m/s²，阈值上限为{round(alarm_threshold, 2)}m/s²，超标量{round(round(spm_lr, 2) - round(alarm_threshold, 2), 2)}m/s²" if alarm_level else None

    results = {
        "type": "轴承磨损低冲能量",
        "energy": round(spm_lr, 4),
        "alarm_threshold": alarm_threshold,
        "alarm_level": alarm_level,
        "alarm_message": alarm_message,
    }
    return results
