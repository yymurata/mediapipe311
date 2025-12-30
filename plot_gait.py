import argparse
import json
import os

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns

if os.environ.get("DISPLAY", "") == "":
    matplotlib.use("Agg")

import matplotlib.pyplot as plt

def calculate_hip_angle_inverted_vec(shoulder_x, shoulder_y, hip_x, hip_y, knee_x, knee_y, is_moving_right):
    """
    股関節の角度を計算（伸展をプラス、屈曲をマイナスに設定）
    - 基準: 直立 = 180度
    - 伸展（後）: 180度 + 開き角 (例: 185度)
    - 屈曲（前）: 180度 - 開き角 (例: 170度)
    """
    v_trunk_x = hip_x - shoulder_x
    v_trunk_y = hip_y - shoulder_y
    v_thigh_x = knee_x - hip_x
    v_thigh_y = knee_y - hip_y

    dot = v_trunk_x * v_thigh_x + v_trunk_y * v_thigh_y
    mag_trunk = np.hypot(v_trunk_x, v_trunk_y)
    mag_thigh = np.hypot(v_thigh_x, v_thigh_y)
    denom = mag_trunk * mag_thigh

    cos_val = np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    theta = np.degrees(np.arccos(cos_val))

    cross = v_trunk_x * v_thigh_y - v_trunk_y * v_thigh_x
    if is_moving_right:
        is_extension = cross > 0
    else:
        is_extension = cross < 0

    angles = np.where(is_extension, 180.0 + theta, 180.0 - theta)
    return np.where(denom == 0, 180.0, angles)

# 膝用の計算（変更なし）
def calculate_knee_angle_vec(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y):
    ba_x, ba_y = hip_x - knee_x, hip_y - knee_y
    bc_x, bc_y = ankle_x - knee_x, ankle_y - knee_y
    dot = ba_x * bc_x + ba_y * bc_y
    mag_ba = np.hypot(ba_x, ba_y)
    mag_bc = np.hypot(bc_x, bc_y)
    denom = mag_ba * mag_bc
    cos_val = np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_val))
    return np.where(denom == 0, 180.0, angles)

def calculate_vector_angle_deg(x1, y1, x2, y2):
    return np.degrees(np.arctan2(y2 - y1, x2 - x1))

def calculate_horizontal_signed_angle_deg(x1, y1, x2, y2):
    dx = x2 - x1
    dy = y2 - y1
    return -np.degrees(np.arctan2(dy, np.abs(dx)))

def calculate_head_posture_angle_deg(x1, y1, x2, y2, is_moving_right):
    dx = x2 - x1
    dy = y2 - y1
    angle_from_up = np.degrees(np.arctan2(dx, -dy))
    if is_moving_right:
        return -angle_from_up
    return angle_from_up

def calculate_head_tilt_relative_deg(
    torso_ax, torso_ay, torso_bx, torso_by,
    head_cx, head_cy, head_dx, head_dy,
    is_moving_right
):
    v1x, v1y = torso_bx - torso_ax, torso_by - torso_ay
    v2x, v2y = head_dx - head_cx, head_dy - head_cy
    dot = v1x * v2x + v1y * v2y
    mag1 = np.hypot(v1x, v1y)
    mag2 = np.hypot(v2x, v2y)
    denom = mag1 * mag2
    cos_val = np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    unsigned = np.degrees(np.arccos(cos_val))
    forward_x = 1.0 if is_moving_right else -1.0
    forward_dot = v2x * forward_x
    signed = np.where(forward_dot > 0, -unsigned, unsigned)
    return np.where(denom == 0, 0.0, signed)

def calculate_line_angle_deg(ax, ay, bx, by, cx, cy, dx, dy):
    v1x, v1y = bx - ax, by - ay
    v2x, v2y = dx - cx, dy - cy
    dot = v1x * v2x + v1y * v2y
    mag1 = np.hypot(v1x, v1y)
    mag2 = np.hypot(v2x, v2y)
    denom = mag1 * mag2
    cos_val = np.divide(dot, denom, out=np.zeros_like(dot), where=denom != 0)
    cos_val = np.clip(cos_val, -1.0, 1.0)
    angles = np.degrees(np.arccos(cos_val))
    return np.where(denom == 0, 180.0, angles)

def detect_unstable_frames(values, k=3.0):
    med = np.median(values)
    mad = np.median(np.abs(values - med))
    thresh = med + k * mad
    return values > thresh

def summarize_unstable_rates(speed_map, k=3.0):
    rates = {}
    for name, values in speed_map.items():
        mask = detect_unstable_frames(values, k=k)
        rates[name] = mask.mean() * 100.0
    return rates

def smooth_series(values, window=5):
    if window <= 1:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")

def apply_time_buffer(mask, times, buffer_sec):
    if buffer_sec <= 0:
        return mask
    expanded = mask.copy()
    selected_times = times[mask]
    if selected_times.size == 0:
        return expanded
    for t in selected_times:
        expanded |= (times >= t - buffer_sec) & (times <= t + buffer_sec)
    return expanded

def build_unstable_mask(points, times, k, buffer_sec):
    speed_list = []
    neg_mask = np.zeros(times.shape, dtype=bool)
    for x, y in points:
        speed_list.append(np.hypot(np.diff(x), np.diff(y)))
        neg_mask |= (x < 0) | (y < 0)
    speed_avg = sum(speed_list) / len(speed_list)
    unstable = detect_unstable_frames(speed_avg, k=k)
    unstable = np.concatenate([[False], unstable])
    unstable |= neg_mask
    unstable = apply_time_buffer(unstable, times, buffer_sec)
    return unstable

def mask_to_segments(times, mask):
    segments = []
    in_segment = False
    start_t = None
    for t, is_bad in zip(times, mask):
        if is_bad and not in_segment:
            start_t = t
            in_segment = True
        elif not is_bad and in_segment:
            segments.append((start_t, t))
            in_segment = False
    if in_segment:
        segments.append((start_t, times[-1]))
    return segments

def select_smoothing_window(
    values,
    times,
    stable_mask,
    peak_window,
    peak_prominence,
    max_window=11,
    cv_threshold=0.3,
    min_window=5
):
    best_window = min_window
    best_cv = None
    best_smooth = smooth_series(values, window=best_window)

    for window in range(min_window, max_window + 1, 2):
        smooth_vals = smooth_series(values, window=window)
        peak_idx, _ = find_contacts_between_peaks(
            smooth_vals,
            times,
            stable_mask,
            min_interval_sec=0.2,
            peak_prominence=peak_prominence,
            peak_window=peak_window
        )
        if len(peak_idx) < 3:
            continue
        intervals = np.diff(times[peak_idx])
        mean = intervals.mean()
        if mean == 0:
            continue
        cv = intervals.std() / mean
        if best_cv is None or cv < best_cv:
            best_cv = cv
            best_window = window
            best_smooth = smooth_vals

    if best_cv is None or best_cv <= cv_threshold:
        return best_window, best_cv, best_smooth
    return best_window, best_cv, best_smooth

def build_peak_window_candidates(times, default_window):
    if times.size < 3:
        return [default_window]
    dt = np.median(np.diff(times))
    if dt <= 0:
        return [default_window]
    min_sec = 0.15
    max_sec = 1.2
    min_window = max(5, int(round(min_sec / dt)))
    max_window = max(min_window, int(round(max_sec / dt)))
    max_window = min(max_window, 101)
    if min_window % 2 == 0:
        min_window += 1
    if max_window % 2 == 0:
        max_window -= 1
    if max_window < min_window:
        return [default_window]
    return list(range(min_window, max_window + 1, 2))

def interval_cv_for_peaks(values, times, mask, peak_window, peak_prominence):
    peak_max = find_local_extrema(
        values, times, mask,
        min_interval_sec=0.2,
        peak_prominence=peak_prominence,
        peak_window=peak_window,
        find_max=True
    )
    peak_min = find_local_extrema(
        values, times, mask,
        min_interval_sec=0.2,
        peak_prominence=peak_prominence,
        peak_window=peak_window,
        find_max=False
    )
    best_peaks = peak_max
    intervals = np.diff(times[best_peaks]) if len(best_peaks) >= 2 else np.array([])
    best_cv = None
    if intervals.size >= 2:
        mean = intervals.mean()
        best_cv = intervals.std() / mean if mean != 0 else None
    alt_intervals = np.diff(times[peak_min]) if len(peak_min) >= 2 else np.array([])
    if alt_intervals.size >= 2:
        mean = alt_intervals.mean()
        alt_cv = alt_intervals.std() / mean if mean != 0 else None
        if best_cv is None or (alt_cv is not None and alt_cv < best_cv):
            best_cv = alt_cv
    return best_cv

def select_peak_window(series_list, times, peak_prominence, default_window):
    candidates = build_peak_window_candidates(times, default_window)
    best_window = default_window
    best_cv = None
    for window in candidates:
        cvs = []
        for values, mask in series_list:
            cv = interval_cv_for_peaks(values, times, mask, window, peak_prominence)
            if cv is not None:
                cvs.append(cv)
        if not cvs:
            continue
        median_cv = float(np.median(cvs))
        if best_cv is None or median_cv < best_cv:
            best_cv = median_cv
            best_window = window
    return best_window, candidates, best_cv

def select_period_smoothing_window(
    series_list,
    times,
    peak_window,
    min_window=5,
    max_window=21
):
    candidates = list(range(min_window, max_window + 1, 2))
    best_window = min_window
    best_cv = None
    for window in candidates:
        cvs = []
        for values, mask, prominence in series_list:
            smooth_vals = smooth_series(values, window=window)
            cv = interval_cv_for_peaks(smooth_vals, times, mask, peak_window, prominence)
            if cv is not None:
                cvs.append(cv)
        if not cvs:
            continue
        median_cv = float(np.median(cvs))
        if best_cv is None or median_cv < best_cv:
            best_cv = median_cv
            best_window = window
    return best_window, candidates, best_cv

def estimate_period_autocorr(values, times, min_period_sec=0.3, max_period_sec=3.0):
    if values.size < 3:
        return None, None
    dt = np.median(np.diff(times))
    if dt <= 0:
        return None, None
    min_lag = max(1, int(round(min_period_sec / dt)))
    max_lag = int(round(max_period_sec / dt))
    max_lag = min(max_lag, values.size - 1)
    if max_lag <= min_lag:
        return None, None

    centered = values - np.mean(values)
    denom = np.dot(centered, centered)
    if denom == 0:
        return None, None
    corr_full = np.correlate(centered, centered, mode="full")
    corr = corr_full[corr_full.size // 2 :] / denom
    segment = corr[min_lag : max_lag + 1]
    if segment.size < 3:
        return None, None

    peaks = []
    for i in range(1, segment.size - 1):
        if segment[i] >= segment[i - 1] and segment[i] >= segment[i + 1]:
            peaks.append(i)
    if not peaks:
        return None, None
    best = max(peaks, key=lambda idx: segment[idx])
    lag = min_lag + best
    return lag * dt, float(segment[best])

def estimate_period_fft(values, times, min_period_sec=0.3, max_period_sec=3.0):
    if values.size < 3:
        return None, None
    dt = np.median(np.diff(times))
    if dt <= 0:
        return None, None
    centered = values - np.mean(values)
    window = np.hanning(centered.size)
    windowed = centered * window
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(windowed.size, d=dt)
    power = np.abs(spectrum) ** 2

    min_freq = 1.0 / max_period_sec
    max_freq = 1.0 / min_period_sec
    valid = (freqs >= min_freq) & (freqs <= max_freq)
    if not np.any(valid):
        return None, None
    idx = np.argmax(power[valid])
    peak_freq = freqs[valid][idx]
    peak_power = power[valid][idx]
    if peak_freq <= 0:
        return None, None
    return 1.0 / peak_freq, float(peak_power)

def apply_bandpass_fft(values, times, baseline_period, bandwidth=0.3):
    if values.size < 3 or baseline_period is None:
        return values
    if not np.isfinite(values).all():
        idx = np.arange(values.size)
        valid = np.isfinite(values)
        if valid.sum() < 2:
            return values
        values = np.interp(idx, idx[valid], values[valid])
    if os.environ.get("BANDPASS_DEBUG", "") == "1":
        print(
            f"bandpass input: min={np.nanmin(values):.4f}, "
            f"max={np.nanmax(values):.4f}, "
            f"nan={np.isnan(values).sum()}"
        )
    dt = np.median(np.diff(times))
    if dt <= 0:
        return values
    center = 1.0 / baseline_period
    low = center / (1.0 + bandwidth)
    high = center / (1.0 - bandwidth)
    if low <= 0 or high <= 0:
        return values
    fft = np.fft.rfft(values)
    freqs = np.fft.rfftfreq(values.size, d=dt)
    mask = (freqs >= low) & (freqs <= high)
    filtered = np.zeros_like(fft)
    filtered[mask] = fft[mask]
    output = np.fft.irfft(filtered, n=values.size)
    if os.environ.get("BANDPASS_DEBUG", "") == "1":
        print(
            f"bandpass output: min={np.nanmin(output):.4f}, "
            f"max={np.nanmax(output):.4f}, "
            f"nan={np.isnan(output).sum()}"
        )
    return output

def score_peak_params(values, times, mask, peak_window, peak_prominence, baseline_range):
    peaks = find_local_extrema(
        values, times, mask,
        min_interval_sec=0.2,
        peak_prominence=peak_prominence,
        peak_window=peak_window,
        find_max=True
    )
    if len(peaks) < 2:
        return None
    intervals = np.diff(times[peaks])
    total = intervals.size
    if total == 0:
        return None
    lo, hi = baseline_range
    in_range = intervals[(intervals >= lo) & (intervals <= hi)]
    if in_range.size == 0:
        return None
    mean = in_range.mean()
    cv = in_range.std() / mean if mean != 0 else None
    ratio = in_range.size / total
    return {
        "peaks": len(peaks),
        "ratio": ratio,
        "count": in_range.size,
        "cv": cv if cv is not None else float("inf"),
    }

def score_valley_params(values, times, mask, peak_window, peak_prominence, baseline_range, find_max=False):
    valleys = find_local_extrema(
        values, times, mask,
        min_interval_sec=0.2,
        peak_prominence=peak_prominence,
        peak_window=peak_window,
        find_max=find_max
    )
    if len(valleys) < 2:
        return None
    intervals = np.diff(times[valleys])
    total = intervals.size
    if total == 0:
        return None
    lo, hi = baseline_range
    in_range = intervals[(intervals >= lo) & (intervals <= hi)]
    if in_range.size == 0:
        return None
    mean = in_range.mean()
    cv = in_range.std() / mean if mean != 0 else None
    ratio = in_range.size / total
    return {
        "peaks": len(valleys),
        "ratio": ratio,
        "count": in_range.size,
        "cv": cv if cv is not None else float("inf"),
    }

def select_toe_peak_params(
    values,
    times,
    mask,
    baseline_range,
    default_peak_window,
    default_prominence,
    prominences=None,
    min_window=5,
    max_window=21
):
    window_candidates = list(range(min_window, max_window + 1, 2))
    peak_windows = build_peak_window_candidates(times, default_peak_window)
    if prominences is None:
        prominences = [default_prominence, default_prominence * 0.5, default_prominence * 0.25]
    best = None
    best_params = None
    for smooth_w in window_candidates:
        smooth_vals = smooth_series(values, window=smooth_w)
        for peak_w in peak_windows:
            for prom in prominences:
                score = score_peak_params(smooth_vals, times, mask, peak_w, prom, baseline_range)
                if score is None:
                    continue
                candidate = (score["count"], score["ratio"], -score["cv"])
                if best is None or candidate > best:
                    best = candidate
                    best_params = {
                        "smooth_window": smooth_w,
                        "peak_window": peak_w,
                        "prominence": prom,
                        "score": score,
                        "smooth_vals": smooth_vals,
                    }
    return best_params

def select_heel_valley_params(
    values,
    times,
    mask,
    baseline_range,
    default_peak_window,
    default_prominence,
    min_window=5,
    max_window=21
):
    window_candidates = list(range(min_window, max_window + 1, 2))
    peak_windows = build_peak_window_candidates(times, default_peak_window)
    prominences = [default_prominence, default_prominence * 0.5, default_prominence * 0.25]
    best = None
    best_params = None
    for smooth_w in window_candidates:
        smooth_vals = smooth_series(values, window=smooth_w)
        for peak_w in peak_windows:
            for prom in prominences:
                score = score_valley_params(smooth_vals, times, mask, peak_w, prom, baseline_range, find_max=True)
                if score is None:
                    continue
                candidate = (score["count"], score["ratio"], -score["cv"])
                if best is None or candidate > best:
                    best = candidate
                    best_params = {
                        "smooth_window": smooth_w,
                        "peak_window": peak_w,
                        "prominence": prom,
                        "score": score,
                        "smooth_vals": smooth_vals,
                    }
    return best_params

def recompute_valleys_between_peaks(values, peak_idx, stable_mask):
    valleys = []
    if len(peak_idx) < 2:
        return valleys
    for start, end in zip(peak_idx, peak_idx[1:]):
        segment = np.arange(start, end + 1)
        segment = segment[stable_mask[segment]]
        if segment.size == 0:
            continue
        valleys.append(int(segment[np.argmax(values[segment])]))
    return valleys

def prune_peaks_by_min_interval(peak_idx, times, baseline_range):
    if len(peak_idx) < 2 or baseline_range is None:
        return peak_idx, 0
    lo, hi = baseline_range
    center = (lo + hi) / 2.0
    peaks = list(sorted(set(int(i) for i in peak_idx)))
    removed = 0

    def score(peaks_list):
        if len(peaks_list) < 2:
            return 0, float("inf")
        intervals = np.diff(times[peaks_list])
        in_range = intervals[(intervals >= lo) & (intervals <= hi)]
        count = in_range.size
        dev = float(np.sum(np.abs(in_range - center))) if count else float("inf")
        return count, dev

    while len(peaks) >= 2:
        intervals = np.diff(times[peaks])
        short_idx = np.where(intervals < lo)[0]
        if short_idx.size == 0:
            break
        i = int(short_idx[0])
        cand_left = peaks[:i] + peaks[i + 1:]
        cand_right = peaks[:i + 1] + peaks[i + 2:]
        score_left = score(cand_left)
        score_right = score(cand_right)
        if score_left > score_right:
            peaks = cand_left
        else:
            peaks = cand_right
        removed += 1

    return peaks, removed

def find_contacts_between_peaks(
    values,
    times,
    stable_mask,
    min_interval_sec=0.2,
    peak_prominence=0.01,
    peak_window=10
):
    peak_idx = []
    last_time = None
    for i in range(1, len(values) - 1):
        if not stable_mask[i]:
            continue
        if values[i] <= values[i - 1] and values[i] <= values[i + 1]:
            start = max(0, i - peak_window)
            end = min(len(values), i + peak_window + 1)
            local_max = values[start:end].max()
            if (local_max - values[i]) < peak_prominence:
                continue
            if last_time is None or (times[i] - last_time) >= min_interval_sec:
                peak_idx.append(i)
                last_time = times[i]

    contact_idx = []
    for start, end in zip(peak_idx, peak_idx[1:]):
        segment = np.arange(start, end + 1)
        segment = segment[stable_mask[segment]]
        if segment.size == 0:
            continue
        max_i = segment[np.argmax(values[segment])]
        contact_idx.append(int(max_i))

    return peak_idx, contact_idx

def find_contacts_with_fallback(
    values,
    times,
    stable_mask,
    min_interval_sec,
    peak_prominence,
    peak_window
):
    for scale in (1.0, 0.5, 0.25, 0.0):
        prom = peak_prominence * scale
        peak_idx, contact_idx = find_contacts_between_peaks(
            values,
            times,
            stable_mask,
            min_interval_sec=min_interval_sec,
            peak_prominence=prom,
            peak_window=peak_window
        )
        if peak_idx:
            return peak_idx, contact_idx, prom
    return [], [], 0.0

def find_local_extrema(
    values,
    times,
    stable_mask,
    min_interval_sec,
    peak_prominence,
    peak_window,
    find_max=True
):
    peak_idx = []
    last_time = None
    for i in range(1, len(values) - 1):
        if not stable_mask[i]:
            continue
        if find_max:
            is_peak = values[i] >= values[i - 1] and values[i] >= values[i + 1]
        else:
            is_peak = values[i] <= values[i - 1] and values[i] <= values[i + 1]
        if not is_peak:
            continue
        start = max(0, i - peak_window)
        end = min(len(values), i + peak_window + 1)
        if find_max:
            local_min = values[start:end].min()
            prominence = values[i] - local_min
        else:
            local_max = values[start:end].max()
            prominence = local_max - values[i]
        if prominence < peak_prominence:
            continue
        if last_time is None or (times[i] - last_time) >= min_interval_sec:
            peak_idx.append(i)
            last_time = times[i]
    return peak_idx

def calculate_slope(values, times):
    dt = np.diff(times)
    dy = np.diff(values)
    slope = np.divide(dy, dt, out=np.zeros_like(dy), where=dt != 0)
    return np.concatenate([[0.0], slope])

def filter_contacts_by_composite_conditions(
    contact_idx,
    toe_peak_idx,
    toe_valley_idx,
    heel_valley_idx,
    toe_y,
    heel_y,
    times,
    period_sec
):
    toe_peak_idx = np.array(toe_peak_idx, dtype=int)
    toe_valley_idx = np.array(toe_valley_idx, dtype=int)
    heel_valley_idx = np.array(heel_valley_idx, dtype=int)
    kept = []
    if toe_peak_idx.size == 0 or toe_valley_idx.size == 0 or heel_valley_idx.size == 0:
        return kept, {}
    if period_sec is None:
        return kept, {}

    for peak_pos, peak_i in enumerate(toe_peak_idx):
        if peak_pos >= toe_valley_idx.size:
            break
        next_valleys = toe_valley_idx[toe_valley_idx > peak_i]
        if next_valleys.size == 0:
            continue
        within_period = next_valleys[
            (times[next_valleys] - times[peak_i]) <= period_sec
        ]
        if within_period.size == 0:
            continue
        toe_valley_i = within_period[0]

        segment = np.arange(peak_i + 1, toe_valley_i + 1)
        if segment.size == 0:
            continue
        heel_candidates = heel_valley_idx[
            (heel_valley_idx >= segment[0]) & (heel_valley_idx <= segment[-1])
        ]
        if heel_candidates.size == 0:
            continue
        best_idx = int(heel_candidates[np.argmax(heel_y[heel_candidates])])
        seg_times = times[segment]
        seg_vals = heel_y[segment]
        dt = np.diff(seg_times)
        dy = np.diff(seg_vals)
        slopes = np.divide(dy, dt, out=np.zeros_like(dy), where=dt != 0)
        slopes = smooth_series(slopes, window=5) if slopes.size >= 5 else slopes
        local_pos = np.where(segment == best_idx)[0]
        if local_pos.size == 0 or local_pos[0] == 0:
            kept.append(best_idx)
            continue
        valley_pos = int(local_pos[0])
        pre_slopes = slopes[:valley_pos]
        if pre_slopes.size == 0:
            kept.append(best_idx)
            continue
        max_pos = int(np.argmax(pre_slopes))
        decel_idx = None
        for j in range(max_pos + 1, pre_slopes.size):
            if pre_slopes[j] < pre_slopes[j - 1]:
                decel_idx = segment[j]
                break
        if decel_idx is not None:
            if decel_idx <= peak_i:
                kept.append(int(peak_i))
            else:
                kept.append(int(decel_idx))
        else:
            kept.append(best_idx)

    return kept, {}

# === メイン処理 ===
parser = argparse.ArgumentParser(description="CSVの歩行解析データを可視化します。")
parser.add_argument("csv", nargs="?", default=os.environ.get("CSV_FILE", ""), help="入力CSVファイルパス")
parser.add_argument("--batch-config", default="", help="batch_config.json を使ってCSVを自動選択")
parser.add_argument("--output-dir", default=os.environ.get("OUTPUT_DIR", "output"), help="gait_analysis_batch.py の出力先")
parser.add_argument("--save", action="store_true", help="グラフをファイルに保存する")
parser.add_argument(
    "--save-dir",
    default=os.environ.get("PLOT_OUTPUT_DIR", os.environ.get("OUTPUT_DIR", "output")),
    help="グラフの保存先フォルダ",
)
parser.add_argument("--save-formats", default="png", help="保存形式 (例: png,pdf)")
parser.add_argument("--no-show", action="store_true", help="グラフ表示を行わない")
parser.add_argument("--start-sec", type=float, default=None, help="表示開始秒（グラフのX軸）")
parser.add_argument("--end-sec", type=float, default=None, help="表示終了秒（グラフのX軸）")
parser.add_argument(
    "--unstable-k",
    type=float,
    default=float(os.environ.get("UNSTABLE_K", "7.0")),
    help="不安定判定の閾値係数 (median + k * MAD)"
)
parser.add_argument(
    "--unstable-buffer-ms",
    type=float,
    default=float(os.environ.get("UNSTABLE_BUFFER_MS", "50")),
    help="不安定区間の前後に追加で除外する時間(ms)"
)
parser.add_argument(
    "--peak-prominence",
    type=float,
    default=float(os.environ.get("PEAK_PROMINENCE", "0.01")),
    help="遊脚ピーク検出の最小プロミネンス (正規化座標)"
)
parser.add_argument(
    "--peak-window",
    type=int,
    default=int(os.environ.get("PEAK_WINDOW", "10")),
    help="ピーク判定の近傍ウィンドウ(フレーム)"
)
parser.add_argument(
    "--no-composite-heel",
    "--no-use-toe-for-heel",
    action="store_false",
    dest="use_composite_heel",
    help="踵接地の複合条件を使わない"
)
parser.add_argument(
    "--toe-after-sec",
    type=float,
    default=float(os.environ.get("TOE_AFTER_SEC", "0.4")),
    help="つま先ピーク後に踵接地が起きる想定の最大時間(秒)"
)
parser.add_argument(
    "--drop-window-sec",
    type=float,
    default=float(os.environ.get("DROP_WINDOW_SEC", "0.08")),
    help="踵/足首の急降下を確認する時間幅(秒)"
)
parser.add_argument(
    "--drop-quantile",
    type=float,
    default=float(os.environ.get("DROP_QUANTILE", "0.8")),
    help="急降下のしきい値に使う分位点(大きいほど厳しい)"
)
parser.set_defaults(use_composite_heel=True)
args = parser.parse_args()

def csv_from_config(config, output_dir):
    video_path = config.get("input_video", "")
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    start_sec = config.get("start_sec", 0)
    end_sec = config.get("end_sec", None)
    time_suffix = f"_{start_sec}s-{end_sec if end_sec else 'end'}s"
    return os.path.join(output_dir, f"{base_name}{time_suffix}_gait.csv")

csv_list = []
if args.batch_config:
    if not os.path.exists(args.batch_config):
        print(f"エラー: {args.batch_config} が見つかりません。")
        exit()
    try:
        with open(args.batch_config, "r", encoding="utf-8") as f:
            batch_list = json.load(f)
    except json.JSONDecodeError:
        print("エラー: JSONファイルの形式が正しくありません。")
        exit()
    csv_list = []
    for config in batch_list:
        analysis_start = config.get("analysis_start_sec")
        analysis_end = config.get("analysis_end_sec")
        display_start = config.get("display_start_sec", analysis_start)
        display_end = config.get("display_end_sec", analysis_end)
        csv_list.append(
            {
                "csv_path": csv_from_config(config, args.output_dir),
                "display_start_sec": display_start,
                "display_end_sec": display_end,
                "analysis_start_sec": analysis_start,
                "analysis_end_sec": analysis_end,
            }
        )
else:
    if not args.csv:
        print("エラー: 入力CSVが指定されていません。引数またはCSV_FILE環境変数で指定してください。")
        exit()
    csv_list = [
        {
            "csv_path": args.csv,
            "display_start_sec": None,
            "display_end_sec": None,
        }
    ]

for item in csv_list:
    csv_path = item["csv_path"]
    if not os.path.exists(csv_path):
        print(f"エラー: {csv_path} が見つかりません。")
        continue
    print("\n" + "=" * 60)
    print(f"解析対象: {csv_path}")
    print("=" * 60)

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # 時間列
    time_col = 'time_sec'
    if time_col not in df.columns:
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        else:
            time_col = df.columns[0]

    analysis_start = item.get("analysis_start_sec")
    analysis_end = item.get("analysis_end_sec")
    if analysis_start is not None or analysis_end is not None:
        mask = pd.Series(True, index=df.index)
        if analysis_start is not None:
            mask &= df[time_col] >= analysis_start
        if analysis_end is not None:
            mask &= df[time_col] <= analysis_end
        df = df[mask].reset_index(drop=True)
        if df.empty:
            print("エラー: 解析区間にデータがありません。")
            continue
        print(
            f"解析区間: {analysis_start if analysis_start is not None else '-'}"
            f" 〜 {analysis_end if analysis_end is not None else '-'}"
        )

    # 進行方向判定
    start_x = df['left_hip_x'].iloc[0]
    end_x = df['left_hip_x'].iloc[-1]
    is_moving_right = True
    direction_text = "右(Right)"
    if end_x < start_x:
        is_moving_right = False
        direction_text = "左(Left)"
    print("== 進行方向判定 ==")
    print(f"判定: 被験者は【{direction_text}】に向かって歩いています。")

    # --- 角度計算（再反転版を使用） ---
    print("== 関節角度計算 ==")
    print("関節角度（設定変更版）を計算中...")

    print("== 追加特徴量計算 (足関節/視線/頭部) ==")
    try:
        df['left_hip_angle'] = calculate_hip_angle_inverted_vec(
            df['left_shoulder_x'].to_numpy(), df['left_shoulder_y'].to_numpy(),
            df['left_hip_x'].to_numpy(), df['left_hip_y'].to_numpy(),
            df['left_knee_x'].to_numpy(), df['left_knee_y'].to_numpy(),
            is_moving_right
        )
        df['left_hip_angle'] = (df['left_hip_angle'] - 180.0) * -1.0

        df['right_hip_angle'] = calculate_hip_angle_inverted_vec(
            df['right_shoulder_x'].to_numpy(), df['right_shoulder_y'].to_numpy(),
            df['right_hip_x'].to_numpy(), df['right_hip_y'].to_numpy(),
            df['right_knee_x'].to_numpy(), df['right_knee_y'].to_numpy(),
            is_moving_right
        )
        df['right_hip_angle'] = (df['right_hip_angle'] - 180.0) * -1.0

        df['left_knee_angle'] = calculate_knee_angle_vec(
            df['left_hip_x'].to_numpy(), df['left_hip_y'].to_numpy(),
            df['left_knee_x'].to_numpy(), df['left_knee_y'].to_numpy(),
            df['left_ankle_x'].to_numpy(), df['left_ankle_y'].to_numpy()
        )
        df['left_knee_angle'] = (df['left_knee_angle'] - 180.0) * -1.0

        df['right_knee_angle'] = calculate_knee_angle_vec(
            df['right_hip_x'].to_numpy(), df['right_hip_y'].to_numpy(),
            df['right_knee_x'].to_numpy(), df['right_knee_y'].to_numpy(),
            df['right_ankle_x'].to_numpy(), df['right_ankle_y'].to_numpy()
        )
        df['right_knee_angle'] = (df['right_knee_angle'] - 180.0) * -1.0

    except KeyError as e:
        print(f"エラー: データ列 {e} が不足しています。")
        continue

    try:
        df['left_ankle_angle'] = calculate_line_angle_deg(
            df['left_knee_x'].to_numpy(), df['left_knee_y'].to_numpy(),
            df['left_ankle_x'].to_numpy(), df['left_ankle_y'].to_numpy(),
            df['left_heel_x'].to_numpy(), df['left_heel_y'].to_numpy(),
            df['left_foot_index_x'].to_numpy(), df['left_foot_index_y'].to_numpy()
        )

        df['right_ankle_angle'] = calculate_line_angle_deg(
            df['right_knee_x'].to_numpy(), df['right_knee_y'].to_numpy(),
            df['right_ankle_x'].to_numpy(), df['right_ankle_y'].to_numpy(),
            df['right_heel_x'].to_numpy(), df['right_heel_y'].to_numpy(),
            df['right_foot_index_x'].to_numpy(), df['right_foot_index_y'].to_numpy()
        )
        df['left_ankle_angle'] = df['left_ankle_angle'] - 90.0
        df['right_ankle_angle'] = df['right_ankle_angle'] - 90.0

        dominant_side = "right" if is_moving_right else "left"
        dominant_label = "右" if is_moving_right else "左"
        dominant_color = "red" if is_moving_right else "blue"

        df['dominant_gaze_angle'] = calculate_horizontal_signed_angle_deg(
            df[f'{dominant_side}_ear_x'].to_numpy(),
            df[f'{dominant_side}_ear_y'].to_numpy(),
            df[f'{dominant_side}_eye_x'].to_numpy(),
            df[f'{dominant_side}_eye_y'].to_numpy()
        )
        df['dominant_head_posture_angle'] = calculate_head_posture_angle_deg(
            df[f'{dominant_side}_hip_x'].to_numpy(),
            df[f'{dominant_side}_hip_y'].to_numpy(),
            df[f'{dominant_side}_shoulder_x'].to_numpy(),
            df[f'{dominant_side}_shoulder_y'].to_numpy(),
            is_moving_right
        )
        df['dominant_head_tilt_angle'] = calculate_head_tilt_relative_deg(
            df[f'{dominant_side}_hip_x'].to_numpy(),
            df[f'{dominant_side}_hip_y'].to_numpy(),
            df[f'{dominant_side}_shoulder_x'].to_numpy(),
            df[f'{dominant_side}_shoulder_y'].to_numpy(),
            df[f'{dominant_side}_shoulder_x'].to_numpy(),
            df[f'{dominant_side}_shoulder_y'].to_numpy(),
            df[f'{dominant_side}_ear_x'].to_numpy(),
            df[f'{dominant_side}_ear_y'].to_numpy(),
            is_moving_right
        )

    except KeyError as e:
        print(f"エラー: 追加解析に必要なデータ列 {e} が不足しています。")
        continue

    dominant_hip_x = df[f'{dominant_side}_hip_x'].to_numpy()
    dominant_hip_y = df[f'{dominant_side}_hip_y'].to_numpy()
    dominant_shoulder_x = df[f'{dominant_side}_shoulder_x'].to_numpy()
    dominant_shoulder_y = df[f'{dominant_side}_shoulder_y'].to_numpy()
    dominant_knee_x = df[f'{dominant_side}_knee_x'].to_numpy()
    dominant_knee_y = df[f'{dominant_side}_knee_y'].to_numpy()
    dominant_heel_x = df[f'{dominant_side}_heel_x'].to_numpy()
    dominant_heel_y = df[f'{dominant_side}_heel_y'].to_numpy()
    dominant_ankle_x = df[f'{dominant_side}_ankle_x'].to_numpy()
    dominant_ankle_y = df[f'{dominant_side}_ankle_y'].to_numpy()
    dominant_foot_x = df[f'{dominant_side}_foot_index_x'].to_numpy()
    dominant_foot_y = df[f'{dominant_side}_foot_index_y'].to_numpy()

    hip_to_knee = np.hypot(dominant_hip_x - dominant_knee_x, dominant_hip_y - dominant_knee_y)
    knee_to_ankle = np.hypot(dominant_knee_x - dominant_ankle_x, dominant_knee_y - dominant_ankle_y)

    print("== 不安定区間の判定 ==")
    times = df[time_col].to_numpy()
    buffer_sec = args.unstable_buffer_ms / 1000.0
    speed_hip = np.hypot(np.diff(dominant_hip_x), np.diff(dominant_hip_y))
    speed_knee = np.hypot(np.diff(dominant_knee_x), np.diff(dominant_knee_y))
    speed_ankle = np.hypot(np.diff(dominant_ankle_x), np.diff(dominant_ankle_y))
    speed_avg = (speed_hip + speed_knee + speed_ankle) / 3.0
    unstable_mask = detect_unstable_frames(speed_avg, k=args.unstable_k)
    unstable_mask = np.concatenate([[False], unstable_mask])
    neg_mask = (
        (dominant_hip_x < 0) | (dominant_hip_y < 0) |
        (dominant_knee_x < 0) | (dominant_knee_y < 0) |
        (dominant_ankle_x < 0) | (dominant_ankle_y < 0)
    )
    unstable_mask = unstable_mask | neg_mask
    unstable_mask = apply_time_buffer(unstable_mask, times, buffer_sec)
    unstable_ratio = unstable_mask.mean() * 100.0

    unstable_hip = build_unstable_mask(
        [
            (dominant_shoulder_x, dominant_shoulder_y),
            (dominant_hip_x, dominant_hip_y),
            (dominant_knee_x, dominant_knee_y),
        ],
        times,
        args.unstable_k,
        buffer_sec
    )
    unstable_knee = build_unstable_mask(
        [
            (dominant_hip_x, dominant_hip_y),
            (dominant_knee_x, dominant_knee_y),
            (dominant_ankle_x, dominant_ankle_y),
        ],
        times,
        args.unstable_k,
        buffer_sec
    )
    unstable_ankle = build_unstable_mask(
        [
            (dominant_knee_x, dominant_knee_y),
            (dominant_ankle_x, dominant_ankle_y),
            (dominant_heel_x, dominant_heel_y),
            (dominant_foot_x, dominant_foot_y),
        ],
        times,
        args.unstable_k,
        buffer_sec
    )

    print("== 平滑化/ピーク検出パラメータ選定 ==")
    stable_mask = ~unstable_mask
    heel_stable_mask = stable_mask & (dominant_heel_x >= 0) & (dominant_heel_y >= 0)
    ankle_stable_mask = (~unstable_ankle) & (dominant_ankle_x >= 0) & (dominant_ankle_y >= 0)
    toe_stable_mask = stable_mask & (dominant_foot_x >= 0) & (dominant_foot_y >= 0)

    knee_values = df[f"{dominant_side}_knee_angle"].to_numpy()
    pre_smooth_window = 5
    peak_window, peak_candidates, peak_cv = select_peak_window(
        [
            (smooth_series(dominant_heel_y, window=pre_smooth_window), heel_stable_mask),
            (smooth_series(dominant_foot_y, window=pre_smooth_window), toe_stable_mask),
            (smooth_series(dominant_ankle_y, window=pre_smooth_window), ankle_stable_mask),
            (smooth_series(knee_values, window=pre_smooth_window), stable_mask),
        ],
        times,
        args.peak_prominence,
        args.peak_window
    )
    if peak_cv is not None and peak_candidates:
        print(
            f"ピークウィンドウ最適化: window={peak_window} "
            f"(candidates={peak_candidates[0]}-{peak_candidates[-1]}), "
            f"median_cv={peak_cv:.3f}"
        )
    else:
        print(f"ピークウィンドウ最適化: window={peak_window}")

    heel_window, heel_cv, smooth_heel_y = select_smoothing_window(
        dominant_heel_y,
        times,
        stable_mask,
        peak_window=peak_window,
        peak_prominence=args.peak_prominence,
        max_window=100,
        cv_threshold=0.3,
        min_window=5
    )
    ankle_window, ankle_cv, smooth_ankle_y = select_smoothing_window(
        dominant_ankle_y,
        times,
        ankle_stable_mask,
        peak_window=peak_window,
        peak_prominence=args.peak_prominence,
        max_window=100,
        cv_threshold=0.3,
        min_window=5
    )
    foot_window, foot_cv, smooth_foot_y = select_smoothing_window(
        dominant_foot_y,
        times,
        stable_mask,
        peak_window=peak_window,
        peak_prominence=args.peak_prominence,
        max_window=100,
        cv_threshold=0.3,
        min_window=5
    )
    if smooth_heel_y.size > times.size:
        smooth_heel_y = smooth_heel_y[: times.size]
    if smooth_ankle_y.size > times.size:
        smooth_ankle_y = smooth_ankle_y[: times.size]
    if smooth_foot_y.size > times.size:
        smooth_foot_y = smooth_foot_y[: times.size]

    print("== 周期推定用データの準備 ==")
    dt = np.median(np.diff(times)) if times.size > 1 else 0.0
    knee_window = int(0.1 / dt) if dt > 0 else 5
    if knee_window < 5:
        knee_window = 5
    if knee_window % 2 == 0:
        knee_window += 1
    knee_values = df[f"{dominant_side}_knee_angle"].to_numpy()
    smooth_knee = smooth_series(knee_values, window=knee_window)

    period_series_raw = [
        (smooth_heel_y, heel_stable_mask, args.peak_prominence),
            (smooth_foot_y, toe_stable_mask, args.peak_prominence),
        (smooth_ankle_y, ankle_stable_mask, args.peak_prominence * 0.5),
        (smooth_knee, stable_mask, args.peak_prominence),
    ]
    period_window, period_candidates, period_cv = select_period_smoothing_window(
        period_series_raw,
        times,
        peak_window,
        min_window=5,
        max_window=21
    )
    if period_cv is not None and period_candidates:
        print(
            f"周期推定 平滑化ウィンドウ: {period_window} フレーム "
            f"(candidates={period_candidates[0]}-{period_candidates[-1]}), "
            f"median_cv={period_cv:.3f}"
        )
    else:
        print(f"周期推定 平滑化ウィンドウ: {period_window} フレーム")

    period_series = [
        ("踵Y", smooth_series(smooth_heel_y, window=period_window), heel_stable_mask, args.peak_prominence),
        ("つま先Y", smooth_series(smooth_foot_y, window=period_window), toe_stable_mask, args.peak_prominence),
        ("足首Y", smooth_series(smooth_ankle_y, window=period_window), ankle_stable_mask, args.peak_prominence * 0.5),
        ("膝角度", smooth_series(smooth_knee, window=period_window), stable_mask, args.peak_prominence),
    ]
    baseline_periods = []
    for _, values, _, _ in period_series:
        period_sec, _ = estimate_period_autocorr(values, times)
        if period_sec is not None:
            baseline_periods.append(period_sec)
        period_sec, _ = estimate_period_fft(values, times)
        if period_sec is not None:
            baseline_periods.append(period_sec)
    baseline_period = float(np.median(baseline_periods)) if baseline_periods else None
    baseline_range = None
    print("== 基準周期の算出 (自己相関/FFT) ==")
    if baseline_period is not None:
        baseline_range = (baseline_period * 0.8, baseline_period * 1.2)
        print(
            f"基準周期: {baseline_period:.3f}s "
            f"(自己相関/FFT中央値, 許容±20%)"
        )
    else:
        print("基準周期: 推定不可")
    min_interval_sec = 0.2

    print("== 周期推定 (自己相関) ==")
    print("周期推定(自己相関):")
    for name, values, mask, prominence in period_series:
        if values.size < 3:
            print(f"  {name}: データ不足")
            continue
        period_sec, score = estimate_period_autocorr(values, times)
        if period_sec is None:
            print(f"  {name}: 推定不可")
            continue
        print(f"  {name}: period={period_sec:.3f}s, peak={score:.3f}")

    print("== 周期推定 (FFT/PSD) ==")
    print("周期推定(FFT/PSD):")
    for name, values, mask, prominence in period_series:
        if values.size < 3:
            print(f"  {name}: データ不足")
            continue
        period_sec, power = estimate_period_fft(values, times)
        if period_sec is None:
            print(f"  {name}: 推定不可")
            continue
        print(f"  {name}: period={period_sec:.3f}s, power={power:.3f}")

    print("== バンドパス適用 (ピーク検出) ==")
    bp_heel = apply_bandpass_fft(smooth_heel_y, times, baseline_period, bandwidth=0.3)
    bp_foot = apply_bandpass_fft(smooth_foot_y, times, baseline_period, bandwidth=0.3)
    bp_ankle = apply_bandpass_fft(smooth_ankle_y, times, baseline_period, bandwidth=0.3)
    print("バンドパス: 基準周期±30% (ピーク検出用)")

    print("== つま先ピーク最適化 ==")
    toe_peak_window = peak_window
    toe_peak_prominence = args.peak_prominence
    smooth_foot_y_peaks = bp_foot
    if baseline_range is not None:
        toe_params = select_toe_peak_params(
            bp_foot,
            times,
            toe_stable_mask,
            baseline_range,
            peak_window,
            args.peak_prominence,
            min_window=5,
            max_window=21
        )
        if toe_params:
            smooth_foot_y_peaks = toe_params["smooth_vals"]
            toe_peak_window = toe_params["peak_window"]
            toe_peak_prominence = toe_params["prominence"]
            score = toe_params["score"]
            print(
                "つま先ピーク最適化: "
                f"smooth={toe_params['smooth_window']}, "
                f"peak_window={toe_peak_window}, "
                f"prominence={toe_peak_prominence:.4f}, "
                f"in_range={score['count']}, "
                f"ratio={score['ratio']:.2f}, "
                f"cv={score['cv']:.3f}"
            )
        else:
            print("つま先ピーク最適化: 該当なし（基準周期内のピーク不足）")
    else:
        print("つま先ピーク最適化: 基準周期なしのためスキップ")

    print("== 踵谷最適化 ==")
    heel_peak_window = peak_window
    heel_peak_prominence = args.peak_prominence
    smooth_heel_y_peaks = bp_heel
    if baseline_range is not None:
        heel_params = select_heel_valley_params(
            bp_heel,
            times,
            heel_stable_mask,
            baseline_range,
            peak_window,
            args.peak_prominence,
            min_window=5,
            max_window=21
        )
        if heel_params:
            smooth_heel_y_peaks = heel_params["smooth_vals"]
            heel_peak_window = heel_params["peak_window"]
            heel_peak_prominence = heel_params["prominence"]
            score = heel_params["score"]
            print(
                "踵谷最適化: "
                f"smooth={heel_params['smooth_window']}, "
                f"peak_window={heel_peak_window}, "
                f"prominence={heel_peak_prominence:.4f}, "
                f"in_range={score['count']}, "
                f"ratio={score['ratio']:.2f}, "
                f"cv={score['cv']:.3f}"
            )
        else:
            print("踵谷最適化: 該当なし（基準周期内の谷不足）")
    else:
        print("踵谷最適化: 基準周期なしのためスキップ")

    print("== 踵/足首/つま先のピーク検出 ==")
    heel_peak_idx, heel_contact_idx_raw, heel_prom = find_contacts_with_fallback(
        smooth_heel_y_peaks,
        times,
        heel_stable_mask,
        min_interval_sec=0.2,
        peak_prominence=heel_peak_prominence,
        peak_window=heel_peak_window
    )
    heel_peak_idx = find_local_extrema(
        smooth_heel_y_peaks,
        times,
        heel_stable_mask,
        min_interval_sec=0.2,
        peak_prominence=heel_peak_prominence,
        peak_window=heel_peak_window,
        find_max=True
    )
    heel_prom = heel_peak_prominence
    if baseline_range is not None and len(heel_peak_idx) >= 2:
        intervals = np.diff(times[heel_peak_idx])
        lo, hi = baseline_range
        if np.any(intervals > hi):
            retry = select_heel_valley_params(
                bp_heel,
                times,
                heel_stable_mask,
                baseline_range,
                peak_window,
                args.peak_prominence,
                min_window=3,
                max_window=21
            )
            if retry:
                smooth_heel_y_peaks = retry["smooth_vals"]
                heel_peak_window = retry["peak_window"]
                heel_peak_prominence = retry["prominence"]
                score = retry["score"]
                print(
                    "踵谷最適化(再探索): "
                    f"smooth={retry['smooth_window']}, "
                    f"peak_window={heel_peak_window}, "
                    f"prominence={heel_peak_prominence:.4f}, "
                    f"in_range={score['count']}, "
                    f"ratio={score['ratio']:.2f}, "
                    f"cv={score['cv']:.3f}"
                )
                heel_peak_idx = find_local_extrema(
                    smooth_heel_y_peaks,
                    times,
                    heel_stable_mask,
                    min_interval_sec=0.2,
                    peak_prominence=heel_peak_prominence,
                    peak_window=heel_peak_window,
                    find_max=True
                )
    toe_peak_idx, toe_valley_idx, toe_prom = find_contacts_with_fallback(
        smooth_foot_y_peaks,
        times,
        toe_stable_mask,
        min_interval_sec=0.2,
        peak_prominence=toe_peak_prominence,
        peak_window=toe_peak_window
    )
    if baseline_range is not None and toe_peak_idx:
        pruned_peaks, removed = prune_peaks_by_min_interval(
            toe_peak_idx, times, baseline_range
        )
        if removed > 0:
            print(
                f"つま先ピーク剪定: removed={removed}, remaining={len(pruned_peaks)}"
            )
        toe_peak_idx = pruned_peaks
        toe_valley_idx = recompute_valleys_between_peaks(
            smooth_foot_y_peaks, toe_peak_idx, toe_stable_mask
        )
        if len(toe_peak_idx) >= 2:
            intervals = np.diff(times[toe_peak_idx])
            lo, hi = baseline_range
            if np.any(intervals > hi):
                retry = select_toe_peak_params(
                    bp_foot,
                    times,
                    toe_stable_mask,
                    baseline_range,
                    peak_window,
                    args.peak_prominence,
                    prominences=[
                        args.peak_prominence,
                        args.peak_prominence * 0.5,
                        args.peak_prominence * 0.25,
                    ],
                    min_window=3,
                    max_window=21
                )
                if retry:
                    smooth_foot_y_peaks = retry["smooth_vals"]
                    toe_peak_window = retry["peak_window"]
                    toe_peak_prominence = retry["prominence"]
                    score = retry["score"]
                    print(
                        "つま先ピーク最適化(再探索): "
                        f"smooth={retry['smooth_window']}, "
                        f"peak_window={toe_peak_window}, "
                        f"prominence={toe_peak_prominence:.4f}, "
                        f"in_range={score['count']}, "
                        f"ratio={score['ratio']:.2f}, "
                        f"cv={score['cv']:.3f}"
                    )
                    toe_peak_idx, toe_valley_idx, toe_prom = find_contacts_with_fallback(
                        smooth_foot_y_peaks,
                        times,
                        toe_stable_mask,
                        min_interval_sec=0.2,
                        peak_prominence=toe_peak_prominence,
                        peak_window=toe_peak_window
                    )
                    pruned_peaks, removed = prune_peaks_by_min_interval(
                        toe_peak_idx, times, baseline_range
                    )
                    if removed > 0:
                        print(
                            f"つま先ピーク剪定: removed={removed}, remaining={len(pruned_peaks)}"
                        )
                    toe_peak_idx = pruned_peaks
                    toe_valley_idx = recompute_valleys_between_peaks(
                        smooth_foot_y_peaks, toe_peak_idx, toe_stable_mask
                    )
    print("== 踵接地(複合条件)の判定 ==")
    heel_contact_idx = heel_contact_idx_raw
    composite_info = {}
    composite_used = False
    if args.use_composite_heel:
        period_sec = None
        if baseline_period is not None:
            period_sec = baseline_period
        heel_contact_idx, composite_info = filter_contacts_by_composite_conditions(
            heel_contact_idx_raw,
            toe_peak_idx,
            toe_valley_idx,
            heel_peak_idx,
            smooth_foot_y_peaks,
            smooth_heel_y_peaks,
            times,
            period_sec
        )
        composite_used = True
        if not heel_contact_idx:
            print("踵接地(複合条件)が検出できませんでした。")
        else:
            contact_times = [f"{times[i]:.3f}" for i in heel_contact_idx]
            print(
                "踵接地(複合条件) 検出: "
                f"{len(heel_contact_idx)}件 "
                f"times=[{', '.join(contact_times)}]"
            )
    ankle_peak_idx, ankle_contact_idx, ankle_prom = find_contacts_with_fallback(
        bp_ankle,
        times,
        ankle_stable_mask,
        min_interval_sec=min_interval_sec,
        peak_prominence=args.peak_prominence,
        peak_window=peak_window
    )
    print(
        f"ピーク数: 踵谷={len(heel_peak_idx)}, つま先山={len(toe_peak_idx)}, 足首山={len(ankle_peak_idx)}"
    )
    if heel_peak_idx:
        heel_times = ", ".join(f"{times[i]:.3f}" for i in heel_peak_idx)
        print(f"踵谷時刻: [{heel_times}]")
    if toe_peak_idx:
        toe_times = ", ".join(f"{times[i]:.3f}" for i in toe_peak_idx)
        print(f"つま先山時刻: [{toe_times}]")
    if ankle_peak_idx:
        ankle_times = ", ".join(f"{times[i]:.3f}" for i in ankle_peak_idx)
        print(f"足首山時刻: [{ankle_times}]")

    ankle_fallback_idx = []
    for i in ankle_contact_idx:
        t = times[i]
        if not any(abs(times[h] - t) <= 0.05 for h in heel_contact_idx):
            ankle_fallback_idx.append(i)

    stable_hip_to_knee = hip_to_knee[stable_mask]
    stable_knee_to_ankle = knee_to_ankle[ankle_stable_mask]

    print(
        f"優位側({dominant_label})の距離: 腰→膝 mean={hip_to_knee.mean():.4f}, median={np.median(hip_to_knee):.4f} | "
        f"膝→足首 mean={knee_to_ankle.mean():.4f}, median={np.median(knee_to_ankle):.4f}"
    )
    if stable_hip_to_knee.size == 0 or stable_knee_to_ankle.size == 0:
        print("安定区間が不足しているため、安定区間の距離統計を計算できません。")
    else:
        print(
            f"安定区間の距離: 腰→膝 mean={stable_hip_to_knee.mean():.4f}, "
            f"median={np.median(stable_hip_to_knee):.4f} | "
            f"膝→足首 mean={stable_knee_to_ankle.mean():.4f}, "
            f"median={np.median(stable_knee_to_ankle):.4f}"
        )
    print(f"トラッキング不安定フレーム率: {unstable_ratio:.1f}%")
    ankle_valid_ratio = ankle_stable_mask.mean() * 100.0
    ankle_segments = mask_to_segments(times, ankle_stable_mask)
    ankle_durations = [end - start for start, end in ankle_segments]
    if ankle_durations:
        ankle_median = float(np.median(ankle_durations))
        ankle_max = float(np.max(ankle_durations))
        print(
            f"足首安定マスク: 有効率={ankle_valid_ratio:.1f}%, "
            f"連続区間={len(ankle_durations)}件, "
            f"median={ankle_median:.3f}s, max={ankle_max:.3f}s"
        )
    else:
        print(f"足首安定マスク: 有効率={ankle_valid_ratio:.1f}%, 連続区間=0件")
    heel_contact_label = f"{dominant_label} 踵接地"
    if composite_used:
        heel_contact_label = f"{dominant_label} 踵接地(複合条件)"
    print(
        f"接地検出: 踵={len(heel_contact_idx)}件, 足首(補完)={len(ankle_fallback_idx)}件"
    )
    if composite_used and composite_info:
        print("複合条件: つま先山→(推定周期以内の)つま先谷→踵谷")
    heel_cv_text = f"{heel_cv:.3f}" if heel_cv is not None else "n/a"
    ankle_cv_text = f"{ankle_cv:.3f}" if ankle_cv is not None else "n/a"
    foot_cv_text = f"{foot_cv:.3f}" if foot_cv is not None else "n/a"
    print(
        f"平滑化ウィンドウ: 踵={heel_window} (cv={heel_cv_text}), "
        f"足首={ankle_window} (cv={ankle_cv_text}), "
        f"つま先={foot_window} (cv={foot_cv_text})"
    )
    print(
        f"ピーク判定プロミネンス: 踵={heel_prom:.4f}, 足首={ankle_prom:.4f}"
    )

    print("== 歩行周期安定区間の抽出 ==")
    stable_cycle_segments = []
    contact_source = heel_contact_idx
    if not args.use_composite_heel and len(contact_source) < 2 and len(heel_contact_idx_raw) >= 2:
        print("歩行周期安定区間: 踵接地が不足のため、従来の踵接地で計算します。")
        contact_source = heel_contact_idx_raw
    if len(contact_source) >= 2:
        contact_times = times[contact_source]
        cycle_intervals = np.diff(contact_times)
        if baseline_range is not None:
            lo, hi = baseline_range
            stable_dt = (cycle_intervals >= lo) & (cycle_intervals <= hi)
            for i, ok in enumerate(stable_dt):
                if ok:
                    stable_cycle_segments.append((contact_times[i], contact_times[i + 1]))
            if stable_cycle_segments:
                print(
                    f"歩行周期安定区間: {len(stable_cycle_segments)}件, "
                    f"range={lo:.3f}-{hi:.3f}s"
                )
            else:
                print("歩行周期安定区間: 該当なし（基準周期±20%内なし）")
        else:
            median_cycle = np.median(cycle_intervals)
            tol = median_cycle * 0.2
            stable_dt = np.abs(cycle_intervals - median_cycle) <= tol
            for i, ok in enumerate(stable_dt):
                if ok:
                    stable_cycle_segments.append((contact_times[i], contact_times[i + 1]))
            if stable_cycle_segments:
                print(
                    f"歩行周期安定区間: {len(stable_cycle_segments)}件, "
                    f"median={median_cycle:.3f}s, tol=±{tol:.3f}s"
                )
            else:
                print("歩行周期安定区間: 該当なし（±20%以内の周期なし）")
    else:
        print("歩行周期安定区間: 踵接地が不足（2点未満）")
    print("== 周期推定 (ピーク間隔) ==")
    reliable_intervals = []
    print("周期推定(ピーク間隔):")
    for name, values, mask, prominence in period_series:
        peak_max = find_local_extrema(
            values, times, mask,
            min_interval_sec=0.2,
            peak_prominence=prominence,
            peak_window=peak_window,
            find_max=True
        )
        peak_min = find_local_extrema(
            values, times, mask,
            min_interval_sec=0.2,
            peak_prominence=prominence,
            peak_window=peak_window,
            find_max=False
        )
        def intervals_with_cv(peak_list):
            intervals = np.diff(times[peak_list]) if len(peak_list) >= 2 else np.array([])
            if baseline_range is not None and intervals.size > 0:
                lo, hi = baseline_range
                intervals = intervals[(intervals >= lo) & (intervals <= hi)]
            if intervals.size >= 2:
                mean = intervals.mean()
                cv = intervals.std() / mean if mean != 0 else None
                return intervals, cv
            return intervals, None

        best_peaks = peak_max
        intervals, best_cv = intervals_with_cv(peak_max)
        alt_intervals, alt_cv = intervals_with_cv(peak_min)
        if alt_cv is not None and (best_cv is None or alt_cv < best_cv):
            best_peaks = peak_min
            intervals = alt_intervals
            best_cv = alt_cv

        if intervals.size < 2:
            print(f"  {name}: ピーク不足")
            continue
        median = np.median(intervals)
        cv = best_cv if best_cv is not None else float("inf")
        print(f"  {name}: peaks={len(best_peaks)}, median={median:.3f}s, cv={cv:.3f}")
        if cv <= 0.2:
            reliable_intervals.append(median)

    if reliable_intervals:
        consensus = float(np.median(reliable_intervals))
        print(f"共通周期(中央値): {consensus:.3f}s")
    else:
        print("共通周期(中央値): 信頼できる系列が不足")

    per_joint_rates = summarize_unstable_rates(
        {"hip": speed_hip, "knee": speed_knee, "ankle": speed_ankle},
        k=args.unstable_k
    )
    worst_joint = max(per_joint_rates, key=per_joint_rates.get)
    print(
        "関節別の不安定率(%): "
        + ", ".join(f"{k}={v:.1f}" for k, v in per_joint_rates.items())
        + f" | 最も不安定: {worst_joint}"
    )

    # --- グラフ描画 ---
    sns.set_theme(style="whitegrid")
    font_family = os.environ.get("PLOT_FONT_FAMILY", "Noto Sans CJK JP")
    plt.rcParams['font.family'] = font_family

    fig, axes = plt.subplots(6, 1, figsize=(12, 21), sharex=True)

    dominant_style = '-'
    nondominant_style = ':'
    left_style = nondominant_style if is_moving_right else dominant_style
    right_style = dominant_style if is_moving_right else nondominant_style

    # 1. 股関節
    sns.lineplot(
        ax=axes[0],
        data=df,
        x=time_col,
        y='left_hip_angle',
        label='左 股関節',
        color='blue',
        linewidth=2,
        linestyle=left_style
    )
    sns.lineplot(
        ax=axes[0],
        data=df,
        x=time_col,
        y='right_hip_angle',
        label='右 股関節',
        color='red',
        linewidth=2,
        linestyle=right_style
    )
    axes[0].set_title('股関節の角度 (プラス=屈曲/前, マイナス=伸展/後)', fontsize=14)
    axes[0].axhline(0, color='black', linestyle='-', alpha=0.8, label='中立ライン', linewidth=1.5)
    axes[0].set_ylabel('角度 (度)', fontsize=12)

    axes[0].set_ylim(-60, 60)
    axes[0].legend(loc='upper right')

    # 2. 膝関節
    sns.lineplot(
        ax=axes[1],
        data=df,
        x=time_col,
        y='left_knee_angle',
        label='左 膝関節',
        color='blue',
        linewidth=2,
        linestyle=left_style
    )
    sns.lineplot(
        ax=axes[1],
        data=df,
        x=time_col,
        y='right_knee_angle',
        label='右 膝関節',
        color='red',
        linewidth=2,
        linestyle=right_style
    )
    axes[1].set_title('膝関節の角度 (プラス=屈曲, マイナス=伸展)', fontsize=14)
    axes[1].axhline(0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
    axes[1].set_ylabel('角度 (度)', fontsize=12)
    axes[1].set_ylim(-120, 20)
    axes[1].legend(loc='upper right')

    # 3. 足関節
    sns.lineplot(
        ax=axes[2],
        data=df,
        x=time_col,
        y='left_ankle_angle',
        label='左 足関節',
        color='blue',
        linewidth=2,
        linestyle=left_style
    )
    sns.lineplot(
        ax=axes[2],
        data=df,
        x=time_col,
        y='right_ankle_angle',
        label='右 足関節',
        color='red',
        linewidth=2,
        linestyle=right_style
    )
    axes[2].set_title('足関節の角度 (プラス=背屈, マイナス=底屈)', fontsize=14)
    axes[2].axhline(0, color='black', linestyle='-', alpha=0.8, linewidth=1.5)
    axes[2].set_ylabel('角度 (度)', fontsize=12)
    axes[2].set_ylim(-90, 90)
    axes[2].legend(loc='upper right')

    # 4. つま先/踵の高さ（優位側）
    sns.lineplot(
        ax=axes[3],
        data=df,
        x=time_col,
        y=f'{dominant_side}_heel_y',
        label=f'{dominant_label} 踵高さ',
        color=dominant_color,
        alpha=0.7
    )
    sns.lineplot(
        ax=axes[3],
        data=df,
        x=time_col,
        y=f'{dominant_side}_foot_index_y',
        label=f'{dominant_label} つま先高さ',
        color=dominant_color,
        alpha=0.7,
        linestyle=':'
    )
    sns.lineplot(
        ax=axes[3],
        data=df,
        x=time_col,
        y=f'{dominant_side}_ankle_y',
        label=f'{dominant_label} 足首高さ',
        color=dominant_color,
        alpha=0.7,
        linestyle='--'
    )
    axes[3].set_title('つま先/踵の上下動（優位側）', fontsize=14)
    axes[3].set_ylabel('高さ (Y座標)', fontsize=12)
    axes[3].legend(loc='upper right')

    # 5. 平滑化した高さ + 接地/ピーク
    sns.lineplot(
        ax=axes[4],
        x=times,
        y=smooth_heel_y,
        label=f'{dominant_label} 踵高さ(平滑化)',
        color=dominant_color,
        alpha=0.8
    )
    sns.lineplot(
        ax=axes[4],
        x=times,
        y=smooth_foot_y,
        label=f'{dominant_label} つま先高さ(平滑化)',
        color=dominant_color,
        alpha=0.8,
        linestyle=':'
    )
    sns.lineplot(
        ax=axes[4],
        x=times,
        y=smooth_ankle_y,
        label=f'{dominant_label} 足首高さ(平滑化)',
        color=dominant_color,
        alpha=0.8,
        linestyle='--'
    )
    if heel_contact_idx:
        axes[4].scatter(
            times[heel_contact_idx],
            np.full(len(heel_contact_idx), 0.02),
            s=120,
            color='green',
            marker='|',
            linewidths=2,
            transform=axes[4].get_xaxis_transform(),
            label=f'{dominant_label} 踵接地(安定ライン)'
        )
    heel_valley_plot_idx = heel_peak_idx
    if baseline_range is not None and len(heel_peak_idx) >= 2:
        lo, hi = baseline_range
        keep = [heel_peak_idx[0]]
        for i in range(1, len(heel_peak_idx)):
            dt = times[heel_peak_idx[i]] - times[heel_peak_idx[i - 1]]
            if lo <= dt <= hi:
                keep.append(heel_peak_idx[i])
        heel_valley_plot_idx = keep
    if heel_valley_plot_idx:
        axes[4].scatter(
            times[heel_valley_plot_idx],
            smooth_heel_y[heel_valley_plot_idx],
            s=35,
            color='black',
            marker='v',
            label=f'{dominant_label} 踵の谷'
        )
    if ankle_peak_idx:
        axes[4].scatter(
            times[ankle_peak_idx],
            smooth_ankle_y[ankle_peak_idx],
            s=20,
            color='gray',
            marker='^',
            label=None
        )
    if toe_peak_idx:
        axes[4].scatter(
            times[toe_peak_idx],
            smooth_foot_y[toe_peak_idx],
            s=35,
            color='black',
            marker='^',
            label=f'{dominant_label} つま先の山'
        )
    axes[4].set_title('踵/足首の上下動（平滑化）と接地/ピーク', fontsize=14)
    axes[4].set_ylabel('高さ (Y座標)', fontsize=12)
    axes[4].legend(loc='upper right')

    # 6. 視線角度（耳→目） + 上半身の傾き（腰→肩） + 頭の傾き（肩→耳）
    sns.lineplot(
        ax=axes[5],
        data=df,
        x=time_col,
        y='dominant_gaze_angle',
        label=f'{dominant_label} 視線角度(耳→目)',
        color=dominant_color,
        linewidth=2
    )
    sns.lineplot(
        ax=axes[5],
        data=df,
        x=time_col,
        y='dominant_head_posture_angle',
        label=f'{dominant_label} 上半身の傾き(腰→肩)',
        color=dominant_color,
        linewidth=2,
        linestyle='--'
    )
    sns.lineplot(
        ax=axes[5],
        data=df,
        x=time_col,
        y='dominant_head_tilt_angle',
        label=f'{dominant_label} 頭の角度(体幹相対)',
        color=dominant_color,
        linewidth=2,
        linestyle=':'
    )
    axes[5].set_title('視線角度/上半身の傾き（画像座標）', fontsize=14)
    axes[5].axhline(0, color='gray', linestyle=':', alpha=0.5)
    axes[5].set_xlabel('時間 (秒)', fontsize=12)
    axes[5].set_ylabel('角度 (度)', fontsize=12)
    axes[5].legend(loc='upper right')

    if stable_cycle_segments:
        for start_t, end_t in stable_cycle_segments:
            axes[4].hlines(
                0.02,
                start_t,
                end_t,
                color='green',
                linewidth=4,
                alpha=0.6,
                transform=axes[4].get_xaxis_transform()
            )

    unstable_segments_map = {
        0: mask_to_segments(times, unstable_hip),
        1: mask_to_segments(times, unstable_knee),
        2: mask_to_segments(times, unstable_ankle),
        3: mask_to_segments(times, unstable_ankle),
        4: mask_to_segments(times, unstable_ankle),
        5: mask_to_segments(times, unstable_hip),
    }
    for idx, segments in unstable_segments_map.items():
        if not segments:
            continue
        for start_t, end_t in segments:
            axes[idx].axvspan(start_t, end_t, color='gray', alpha=0.15)

    display_start = args.start_sec if args.start_sec is not None else item["display_start_sec"]
    display_end = args.end_sec if args.end_sec is not None else item["display_end_sec"]
    if display_start is not None or display_end is not None:
        for ax in axes:
            ax.set_xlim(left=display_start, right=display_end)

    if display_start is not None or display_end is not None:
        mask = pd.Series(True, index=df.index)
        if display_start is not None:
            mask &= df[time_col] >= display_start
        if display_end is not None:
            mask &= df[time_col] <= display_end
        df_range = df[mask]
        if not df_range.empty:
            def set_ylim(ax, values):
                ymin = values.min()
                ymax = values.max()
                if np.isfinite(ymin) and np.isfinite(ymax):
                    pad = (ymax - ymin) * 0.05
                    if pad == 0:
                        pad = 1.0
                    ax.set_ylim(ymin - pad, ymax + pad)

            set_ylim(axes[0], pd.concat([df_range['left_hip_angle'], df_range['right_hip_angle']]))
            set_ylim(axes[1], pd.concat([df_range['left_knee_angle'], df_range['right_knee_angle']]))
            set_ylim(axes[2], pd.concat([df_range['left_ankle_angle'], df_range['right_ankle_angle']]))
            set_ylim(
                axes[3],
                pd.concat([
                    df_range[f'{dominant_side}_heel_y'],
                    df_range[f'{dominant_side}_foot_index_y'],
                    df_range[f'{dominant_side}_ankle_y']
                ])
            )
            set_ylim(
                axes[4],
                pd.concat([
                    pd.Series(smooth_heel_y, index=df.index)[mask],
                    pd.Series(smooth_ankle_y, index=df.index)[mask]
                ])
            )
            set_ylim(
                axes[5],
                pd.concat([
                    df_range['dominant_gaze_angle'],
                    df_range['dominant_head_posture_angle'],
                    df_range['dominant_head_tilt_angle']
                ])
            )
            ymin, ymax = axes[5].get_ylim()
            if ymin > 0:
                ymin = 0
            if ymax < 0:
                ymax = 0
            if ymin == 0:
                ymin -= 1
            if ymax == 0:
                ymax += 1
            axes[5].set_ylim(ymin, ymax)

    axes[3].invert_yaxis()
    axes[4].invert_yaxis()

    plt.tight_layout()

    if args.save:
        os.makedirs(args.save_dir, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(csv_path))[0]
        formats = [fmt.strip() for fmt in args.save_formats.split(",") if fmt.strip()]
        for fmt in formats:
            output_path = os.path.join(args.save_dir, f"{base_name}_plot.{fmt}")
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            print(f"保存しました: {output_path}")

    if args.no_show or os.environ.get("DISPLAY", "") == "":
        print("グラフ表示はスキップしました。")
    else:
        print("グラフを表示します。")
        plt.show()
