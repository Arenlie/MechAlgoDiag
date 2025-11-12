import numpy as np
from model.feature_calculate import calc_HDS, fft_spectrum, calc_vel_pass_rms, calc_acc_kurtosis, calc_HS, acc2dis


def detect_unbalance_fault(s_signal, acc_signal, fs, fr, notice_th=None, warn_th=None):
    s_signal = np.asarray(s_signal, dtype=float)
    acc_signal = np.asarray(acc_signal, dtype=float)
    if s_signal.ndim != 1:
        raise ValueError("s_signal 必须为一维数组。")
    if fr <= 0 or fs <= 0:
        raise ValueError("fr 与 fs 必须为正。")

    # 1) 频谱与倍频峰值
    f, fft_data = fft_spectrum(s_signal, fs)
    energy_2_5 = calc_HS(fft_data, fr, fs / len(s_signal), 2, 4)
    energy_1 = calc_HS(fft_data, fr, fs / len(s_signal), 1, 1)

    # RMS & 峭度
    rms = calc_vel_pass_rms(acc_signal, fs)
    kurtosis = calc_acc_kurtosis(acc_signal, fs)
    # 报警阈值
    alarm_threshold = 0.6 * rms
    # 条件判断
    cond1 = energy_2_5 < 0.4 * rms
    cond2 = energy_1 > alarm_threshold
    cond3 = kurtosis < 3.5
    cond = bool(cond1 & cond2 & cond3)
    if cond and notice_th <= rms <= warn_th:
        alarm_level = "注意"
    elif cond and rms > warn_th:
        alarm_level = "警告"
    else:
        alarm_level = None
    alarm_message = f"不平衡能量【{alarm_level}级】报警，能量为{round(energy_1, 2)}m/s²，阈值上限为{round(alarm_threshold, 2)}m/s²，超标量{round(round(energy_1, 2) - round(alarm_threshold, 2), 2)}m/s²" if alarm_level else None

    result = {
        "type": "不平衡",
        "energy": round(energy_1, 4),
        "alarm_threshold": alarm_threshold,
        "alarm_level": alarm_level,
        "alarm_message": alarm_message,
    }
    return result

