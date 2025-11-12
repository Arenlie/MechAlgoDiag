import numpy as np
from model.feature_calculate import calc_HDS, fft_spectrum, calc_vel_pass_rms, calc_acc_kurtosis, calc_HS, acc2dis


def detect_misalignment_fault(s_signal, acc_signal, fs, fr, notice_th=None, warn_th=None):
    s_signal = np.asarray(s_signal, dtype=float)
    acc_signal = np.asarray(acc_signal, dtype=float)
    if s_signal.ndim != 1:
        raise ValueError("s_signal 必须为一维数组。")

    # 1) 频谱与倍频峰值
    f, fft_data = fft_spectrum(s_signal, fs)
    energy_1_2 = calc_HS(fft_data, fr, fs / len(s_signal), 2, 1)
    energy_3_6 = calc_HS(fft_data, fr, fs / len(s_signal), 3, 3)

    # RMS & 峭度
    rms = calc_vel_pass_rms(acc_signal, fs)
    kurtosis = calc_acc_kurtosis(acc_signal, fs)
    # 报警阈值
    alarm_threshold = 0.3 * rms
    # 条件判断
    cond1 = energy_1_2 >= alarm_threshold
    cond2 = energy_3_6 < 0.25 * rms
    cond3 = kurtosis < 3.5
    cond = bool(cond1 and cond2 and cond3)
    if cond and notice_th <= rms <= warn_th:
        alarm_level = "注意"
    elif cond and rms > warn_th:
        alarm_level = "警告"
    else:
        alarm_level = None
    alarm_message = f"不对中能量【{alarm_level}级】报警，能量为{round(energy_1_2, 2)}m/s²，阈值上限为{round(alarm_threshold, 2)}m/s²，超标量{round(round(energy_1_2, 2) - round(alarm_threshold, 2), 2)}m/s²" if alarm_level else None
    # print("energy_1_2:", round(energy_1_2, 4))
    result = {
        "type": "不对中",
        "energy": round(energy_1_2, 4),
        "alarm_threshold": round(alarm_threshold, 4),
        "alarm_level": alarm_level,
        "alarm_message": alarm_message,
    }
    return result
