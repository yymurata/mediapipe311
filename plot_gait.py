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

        df['right_hip_angle'] = calculate_hip_angle_inverted_vec(
            df['right_shoulder_x'].to_numpy(), df['right_shoulder_y'].to_numpy(),
            df['right_hip_x'].to_numpy(), df['right_hip_y'].to_numpy(),
            df['right_knee_x'].to_numpy(), df['right_knee_y'].to_numpy(),
            is_moving_right
        )

        df['left_knee_angle'] = calculate_knee_angle_vec(
            df['left_hip_x'].to_numpy(), df['left_hip_y'].to_numpy(),
            df['left_knee_x'].to_numpy(), df['left_knee_y'].to_numpy(),
            df['left_ankle_x'].to_numpy(), df['left_ankle_y'].to_numpy()
        )

        df['right_knee_angle'] = calculate_knee_angle_vec(
            df['right_hip_x'].to_numpy(), df['right_hip_y'].to_numpy(),
            df['right_knee_x'].to_numpy(), df['right_knee_y'].to_numpy(),
            df['right_ankle_x'].to_numpy(), df['right_ankle_y'].to_numpy()
        )

    except KeyError as e:
        print(f"エラー: データ列 {e} が不足しています。")
        continue

    # --- グラフ描画 ---
    sns.set_theme(style="whitegrid")
    font_family = os.environ.get("PLOT_FONT_FAMILY", "Noto Sans CJK JP")
    plt.rcParams['font.family'] = font_family

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)

    # 1. 股関節
    sns.lineplot(ax=axes[0], data=df, x=time_col, y='left_hip_angle', label='左 股関節', color='blue', linewidth=2)
    sns.lineplot(ax=axes[0], data=df, x=time_col, y='right_hip_angle', label='右 股関節', color='red', linewidth=2, linestyle='--')
    axes[0].set_title('股関節の角度 (数値大=伸展/後, 数値小=屈曲/前)', fontsize=14) # タイトル変更
    axes[0].axhline(180, color='gray', linestyle=':', alpha=0.8, label='直立ライン')
    axes[0].set_ylabel('角度 (度)', fontsize=12)

    # Y軸の範囲を調整（値の分布が変わるため）
    # 屈曲=180以下、伸展=180以上になるので、130〜210程度を表示
    axes[0].set_ylim(130, 210)
    axes[0].legend(loc='upper right')

    # 2. 膝関節
    sns.lineplot(ax=axes[1], data=df, x=time_col, y='left_knee_angle', label='左 膝関節', color='blue', linewidth=2)
    sns.lineplot(ax=axes[1], data=df, x=time_col, y='right_knee_angle', label='右 膝関節', color='red', linewidth=2, linestyle='--')
    axes[1].set_title('膝関節の角度 (180=伸展, 小さい=屈曲)', fontsize=14)
    axes[1].axhline(180, color='gray', linestyle=':', alpha=0.5)
    axes[1].set_ylabel('角度 (度)', fontsize=12)
    axes[1].set_ylim(60, 190)
    axes[1].legend(loc='upper right')

    # 3. 足首高さ
    sns.lineplot(ax=axes[2], data=df, x=time_col, y='left_ankle_y', label='左 足首高さ', color='blue', alpha=0.7)
    sns.lineplot(ax=axes[2], data=df, x=time_col, y='right_ankle_y', label='右 足首高さ', color='red', alpha=0.7, linestyle='--')
    axes[2].set_title('足首の上下動 (リズム確認用)', fontsize=14)
    axes[2].set_xlabel('時間 (秒)', fontsize=12)
    axes[2].set_ylabel('高さ (Y座標)', fontsize=12)

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
            set_ylim(axes[2], pd.concat([df_range['left_ankle_y'], df_range['right_ankle_y']]))

    axes[2].invert_yaxis()

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
