from math import isclose, floor, ceil
import numpy as np
from feature_calculate import acc2dis, fft_filter, fft_spectrum


def _is_harmonic(target: float, base: float, tol: float) -> bool:
    """判断 target 是否是 base 的整数倍频"""
    if base == 0:
        return False
    for i in range(1, 10):
        if isclose(target, i * base, abs_tol=tol):
            return True
    return False


def remove_close_peaks(frequencies, amplitudes, min_distance=5):
    """去除相近频率（频率索引差值小于min_distance的，只保留较大幅值）"""
    sorted_idx = np.argsort(amplitudes)[::-1]
    keep = np.ones(len(frequencies), dtype=bool)

    for i in range(len(sorted_idx)):
        if not keep[sorted_idx[i]]:
            continue
        for j in range(i + 1, len(sorted_idx)):
            if abs(frequencies[sorted_idx[i]] - frequencies[sorted_idx[j]]) < min_distance:
                keep[sorted_idx[j]] = False

    return frequencies[keep], amplitudes[keep]


def Speed_Estimate_algorithm(data, fs, num_fre=10, min_fre=0, tol=2):
    """转速估计算法"""
    data_filtered = fft_filter(data, 5, 215, fs)  # 频域滤波

    f, fft_data = fft_spectrum(data_filtered, fs)  # 频谱分析

    # 找出最大的15个峰值
    top_indices = np.argsort(fft_data)[-num_fre:][::-1]
    top_freqs = f[top_indices]
    top_amps = fft_data[top_indices]

    # 删除幅值低于 0.2 的
    mask = top_amps >= min_fre
    top_freqs = top_freqs[mask]
    top_amps = top_amps[mask]

    # 去除相近频率
    top_freqs, top_amps = remove_close_peaks(top_freqs, top_amps)

    # 检查是否有足够的峰值
    if len(top_amps) < 2:
        return 0

    # 如果最大峰远大于第二峰，直接用最大峰估计
    if top_amps[0] / top_amps[1] > 5:
        return top_freqs[0] * 60

    # 寻找最小频率作为基频候选
    min_freq_1 = np.min(top_freqs)
    # 获取min_amp对应的freq
    top_two_indices = np.argsort(top_amps)[-2:][::-1]
    max_amp_freq_1 = top_freqs[top_two_indices[0]]
    max_amp_freq_2 = top_freqs[top_two_indices[1]]

    results = []
    priority_freqs = [min_freq_1, max_amp_freq_1, max_amp_freq_2]

    # 统计各个倍频的支持数
    for min_freq in priority_freqs:
        harmonic_counts = {k: 0 for k in ["half", "exact", "double", "triple", "quadruple"]}
        freq_candidates = {}
        for freq in top_freqs:
            if _is_harmonic(freq, 2 * min_freq, tol):
                harmonic_counts["half"] += 1
                freq_candidates["half"] = 2 * min_freq
            if _is_harmonic(freq, min_freq, tol):
                harmonic_counts["exact"] += 1
                freq_candidates["exact"] = min_freq
            if _is_harmonic(freq, min_freq / 2, tol):
                harmonic_counts["double"] += 1
                freq_candidates["double"] = min_freq / 2
            if _is_harmonic(freq, min_freq / 3, tol):
                harmonic_counts["triple"] += 1
                freq_candidates["triple"] = min_freq / 3
            if _is_harmonic(freq, min_freq / 4, tol):
                harmonic_counts["quadruple"] += 1
                freq_candidates["quadruple"] = min_freq / 4
        # 根据支持数选择
        max_count = max(harmonic_counts.values())
        # 找出所有 count 等于 max_count 的模式
        candidates = [mode for mode, count in harmonic_counts.items() if count == max_count]
        # 优先顺序
        priority_order = ["exact", "half", "double", "triple", "quadruple"]
        # 从优先列表中选出第一个出现在 candidates 里的
        for preferred in priority_order:
            if preferred in candidates:
                best_mode = preferred
                break

        # 保存当前 min_freq 的结果
        results.append({
            "min_freq": min_freq,
            "harmonic_counts": harmonic_counts,
            "freq_candidates": freq_candidates,
            "max_count": max_count,
            "best_mode": best_mode,
            "best_freq_candidate": freq_candidates[best_mode]
        })
        # print("倍频次数", harmonic_counts)

    # 找出所有 max_count 最大的结果
    max_support = max(r["max_count"] for r in results)
    max_results = [r for r in results if r["max_count"] == max_support]

    # 如果存在多个结果，根据 priority_min_freq 优先顺序选择
    for freq in priority_freqs:
        for res in max_results:
            if res["min_freq"] == freq:
                best_result = res
                break
        else:
            continue
        break

    # 从两个 min_freq 的结果中选出支持数最多的那个
    best_mode = best_result['best_mode']
    freq_candidates = best_result['freq_candidates']
    harmonic_counts = best_result['harmonic_counts']
    max_count = best_result['max_count']
    if max_count == 0:
        return 0

    return freq_candidates[best_mode] * 60
