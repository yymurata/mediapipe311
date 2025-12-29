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

def select_smoothing_window(values, times, stable_mask, max_window=11, cv_threshold=0.3, min_window=5):
    best_window = min_window
    best_cv = None
    best_smooth = smooth_series(values, window=best_window)

    for window in range(min_window, max_window + 1, 2):
        smooth_vals = smooth_series(values, window=window)
        peak_idx, _ = find_contacts_between_peaks(smooth_vals, times, stable_mask, min_interval_sec=0.2)
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
    default=float(os.environ.get("UNSTABLE_K", "5.0")),
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
        display_start = config.get("display_start_sec", config.get("start_sec"))
        display_end = config.get("display_end_sec", config.get("end_sec"))
        csv_list.append(
            {
                "csv_path": csv_from_config(config, args.output_dir),
                "display_start_sec": display_start,
                "display_end_sec": display_end,
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

    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]

    # 時間列
    time_col = 'time_sec'
    if time_col not in df.columns:
        if 'timestamp' in df.columns:
            time_col = 'timestamp'
        else:
            time_col = df.columns[0]

    # 進行方向判定
    start_x = df['left_hip_x'].iloc[0]
    end_x = df['left_hip_x'].iloc[-1]
    is_moving_right = True
    direction_text = "右(Right)"
    if end_x < start_x:
        is_moving_right = False
        direction_text = "左(Left)"
    print(f"判定: 被験者は【{direction_text}】に向かって歩いています。")

    # --- 角度計算（再反転版を使用） ---
    print("関節角度（設定変更版）を計算中...")

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

    stable_mask = ~unstable_mask
    ankle_stable_mask = ~unstable_ankle
    heel_window, heel_cv, smooth_heel_y = select_smoothing_window(
        dominant_heel_y,
        times,
        stable_mask,
        max_window=100,
        cv_threshold=0.3,
        min_window=5
    )
    ankle_window, ankle_cv, smooth_ankle_y = select_smoothing_window(
        dominant_ankle_y,
        times,
        ankle_stable_mask,
        max_window=100,
        cv_threshold=0.3,
        min_window=5
    )
    foot_window, foot_cv, smooth_foot_y = select_smoothing_window(
        dominant_foot_y,
        times,
        stable_mask,
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
    heel_stable_mask = stable_mask & (dominant_heel_x >= 0) & (dominant_heel_y >= 0)
    ankle_stable_mask = ankle_stable_mask & (dominant_ankle_x >= 0) & (dominant_ankle_y >= 0)

    heel_peak_idx, heel_contact_idx, heel_prom = find_contacts_with_fallback(
        smooth_heel_y,
        times,
        heel_stable_mask,
        min_interval_sec=0.2,
        peak_prominence=args.peak_prominence,
        peak_window=args.peak_window
    )
    ankle_peak_idx, ankle_contact_idx, ankle_prom = find_contacts_with_fallback(
        smooth_ankle_y,
        times,
        ankle_stable_mask,
        min_interval_sec=0.2,
        peak_prominence=args.peak_prominence,
        peak_window=args.peak_window
    )

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
    print(
        f"接地検出: 踵={len(heel_contact_idx)}件, 足首(補完)={len(ankle_fallback_idx)}件"
    )
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
            smooth_heel_y[heel_contact_idx],
            s=30,
            color='black',
            marker='o',
            label=f'{dominant_label} 踵接地'
        )
    if heel_peak_idx:
        axes[4].scatter(
            times[heel_peak_idx],
            smooth_heel_y[heel_peak_idx],
            s=20,
            color='gray',
            marker='^',
            label=f'{dominant_label} 踵遊脚ピーク'
        )
    if ankle_fallback_idx:
        axes[4].scatter(
            times[ankle_fallback_idx],
            smooth_ankle_y[ankle_fallback_idx],
            s=30,
            color='black',
            marker='x',
            label=f'{dominant_label} 足首接地(補完)'
        )
    if ankle_peak_idx:
        axes[4].scatter(
            times[ankle_peak_idx],
            smooth_ankle_y[ankle_peak_idx],
            s=20,
            color='gray',
            marker='^',
            label=f'{dominant_label} 足首遊脚ピーク'
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
