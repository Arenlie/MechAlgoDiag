import numpy as np
import pandas as pd

from model.feature_calculate import fft_spectrum, calc_vel_pass_rms, calc_acc_kurtosis, calc_HS, acc2dis


def detect_looseness_fault(s_signal, acc_signal, fs, fr, notice_th, warn_th):
    s_signal = np.asarray(s_signal, dtype=float)
    acc_signal = np.asarray(acc_signal, dtype=float)
    if s_signal.ndim != 1:
        raise ValueError("s_signal 必须为一维数组。")
    # if fr <= 0 or fs <= 0:
    #     raise ValueError("fr 与 fs 必须为正。")
    # 1) 频谱与倍频峰值
    f, fft_data = fft_spectrum(s_signal, fs)
    # energy_3_5 = calc_HS(fft_data, fr, fs / len(s_signal), 3, 3)
    energy_3_8 = calc_HS(fft_data, fr, fs / len(s_signal), 3, 6)
    # energy_1_2 = calc_HS(fft_data, fr, fs / len(s_signal), 1, 2)
    # RMS & 峭度
    rms = calc_vel_pass_rms(acc_signal, fs)
    # 报警阈值
    alarm_threshold = 0.3 * rms
    # 条件判断
    cond = energy_3_8 > alarm_threshold
    if cond and notice_th <= rms <= warn_th:
        alarm_level = "注意"
    elif cond and rms > warn_th:
        alarm_level = "警告"
    else:
        alarm_level = None
    alarm_message = f"连接松动能量【{alarm_level}级】报警，能量为{round(energy_3_8, 2)}m/s²，阈值上限为{round(alarm_threshold, 2)}m/s²，超标量{round(round(energy_3_8, 2) - round(alarm_threshold, 2), 2)}m/s²" if alarm_level else None

    result = {
        "type": "连接松动",
        "energy": round(energy_3_8, 4),
        "alarm_threshold": alarm_threshold,
        "alarm_level": alarm_level,
        "alarm_message": alarm_message,
    }
    return result
