import argparse
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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
args = parser.parse_args()

if not args.csv:
    print("エラー: 入力CSVが指定されていません。引数またはCSV_FILE環境変数で指定してください。")
    exit()

if not os.path.exists(args.csv):
    print(f"エラー: {args.csv} が見つかりません。")
    exit()

df = pd.read_csv(args.csv)
df.columns = [c.strip() for c in df.columns]

# 時間列
time_col = 'time_sec'
if time_col not in df.columns:
    if 'timestamp' in df.columns: time_col = 'timestamp'
    else: time_col = df.columns[0]

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
    exit()

# --- グラフ描画 ---
sns.set_theme(style="whitegrid")
plt.rcParams['font.family'] = 'MS Gothic' 

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
axes[2].invert_yaxis()
axes[2].set_title('足首の上下動 (リズム確認用)', fontsize=14)
axes[2].set_xlabel('時間 (秒)', fontsize=12)
axes[2].set_ylabel('高さ (Y座標)', fontsize=12)

plt.tight_layout()
print("グラフを表示します。")
plt.show()
