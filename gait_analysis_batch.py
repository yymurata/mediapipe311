import cv2
import time
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import subprocess
import numpy as np
import json
import os
import shutil

# === 設定 ===
MODEL_PATH = 'pose_landmarker_heavy.task'
CONFIG_FILE = 'batch_config.json' # バッチ処理リスト
SHOW_WINDOW = os.environ.get('SHOW_WINDOW', '1') not in ('0', 'false', 'False')
OUTPUT_DIR = os.environ.get('OUTPUT_DIR', 'output')

def process_video(config):
    video_path = config.get('input_video')
    start_sec = config.get('start_sec', 0)
    end_sec = config.get('end_sec', None)

    if not os.path.exists(video_path):
        print(f"エラー: ファイルが見つかりません -> {video_path}")
        return

    # 出力ファイル名の自動生成 (例: input_5-15s_gait.csv)
    base_name = os.path.splitext(os.path.basename(video_path))[0]
    time_suffix = f"_{start_sec}s-{end_sec if end_sec else 'end'}s"
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_output = os.path.join(OUTPUT_DIR, f"{base_name}{time_suffix}_gait.csv")
    video_output = os.path.join(OUTPUT_DIR, f"{base_name}{time_suffix}_tracked.mp4")

    print(f"\n=== 処理開始: {video_path} ===")
    print(f"区間: {start_sec}秒 〜 {end_sec if end_sec else '最後'}まで")
    print(f"出力: {csv_output}, {video_output}")

    # MediaPipe設定
    base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        output_segmentation_masks=False
    )

    # CSVヘッダー
    header = [
        'frame', 'time_sec',
        'left_shoulder_x', 'left_shoulder_y',
        'left_hip_x', 'left_hip_y',
        'left_knee_x', 'left_knee_y',
        'left_ankle_x', 'left_ankle_y',
        'right_shoulder_x', 'right_shoulder_y',
        'right_hip_x', 'right_hip_y',
        'right_knee_x', 'right_knee_y',
        'right_ankle_x', 'right_ankle_y',
        'left_eye_x', 'left_eye_y',
        'right_eye_x', 'right_eye_y',
        'left_ear_x', 'left_ear_y',
        'right_ear_x', 'right_ear_y',
        'left_heel_x', 'left_heel_y',
        'right_heel_x', 'right_heel_y',
        'left_foot_index_x', 'left_foot_index_y',
        'right_foot_index_x', 'right_foot_index_y'
    ]
    
    with open(csv_output, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"エラー: 動画を開けませんでした -> {video_path}")
            return
        
        # 動画情報の取得
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0: fps = 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # ★開始時間へシーク（ジャンプ）
        if start_sec > 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, start_sec * 1000)

        # FFmpegプロセス起動
        if shutil.which('ffmpeg') is None:
            print("エラー: ffmpeg がインストールされていません。ffmpeg をインストールしてから再実行してください。")
            cap.release()
            return

        command = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{width}x{height}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx265',
            '-crf', '28',
            '-preset', 'fast',
            '-pix_fmt', 'yuv420p',
            video_output
        ]
        
        # 標準エラー出力を抑制したい場合は stderr=subprocess.DEVNULL を追加
        process = subprocess.Popen(command, stdin=subprocess.PIPE)
        f_csv = open(csv_output, 'a', newline='')
        writer = csv.writer(f_csv)

        # CSVに記録するフレーム番号は「0」から始めるか「元の動画のフレーム」にするか
        # ここでは「解析開始地点からの連番」としてカウントします
        processed_frame_count = 0 

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                # 現在の動画時間を取得(秒)
                current_time_sec = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

                # ★終了判定
                if end_sec is not None and current_time_sec > end_sec:
                    print("指定終了時間に到達しました。")
                    break

                # 解析
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
                
                # MediaPipeには「動画上の絶対時刻(ms)」を渡すのが安全
                timestamp_ms = int(current_time_sec * 1000)

                detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

                if detection_result.pose_landmarks:
                    lm = detection_result.pose_landmarks[0]
                    
                    row = [
                        processed_frame_count, # 解析開始からの連番
                        current_time_sec,      # 動画内の絶対時刻
                        lm[11].x, lm[11].y, lm[23].x, lm[23].y, 
                        lm[25].x, lm[25].y, lm[27].x, lm[27].y, 
                        lm[12].x, lm[12].y, lm[24].x, lm[24].y,
                        lm[26].x, lm[26].y, lm[28].x, lm[28].y,
                        lm[2].x, lm[2].y, lm[5].x, lm[5].y,
                        lm[7].x, lm[7].y, lm[8].x, lm[8].y,
                        lm[29].x, lm[29].y, lm[30].x, lm[30].y,
                        lm[31].x, lm[31].y, lm[32].x, lm[32].y
                    ]
                    writer.writerow(row)

                    # 描画
                    h, w, _ = frame.shape
                    # 左（青）
                    cv2.line(frame, (int(lm[11].x*w), int(lm[11].y*h)), (int(lm[23].x*w), int(lm[23].y*h)), (255, 0, 0), 2)
                    cv2.line(frame, (int(lm[23].x*w), int(lm[23].y*h)), (int(lm[25].x*w), int(lm[25].y*h)), (255, 0, 0), 2)
                    cv2.line(frame, (int(lm[25].x*w), int(lm[25].y*h)), (int(lm[27].x*w), int(lm[27].y*h)), (255, 0, 0), 2)
                    # 右（赤）
                    cv2.line(frame, (int(lm[12].x*w), int(lm[12].y*h)), (int(lm[24].x*w), int(lm[24].y*h)), (0, 0, 255), 2)
                    cv2.line(frame, (int(lm[24].x*w), int(lm[24].y*h)), (int(lm[26].x*w), int(lm[26].y*h)), (0, 0, 255), 2)
                    cv2.line(frame, (int(lm[26].x*w), int(lm[26].y*h)), (int(lm[28].x*w), int(lm[28].y*h)), (0, 0, 255), 2)
                    # 眼/耳（点）
                    cv2.circle(frame, (int(lm[2].x*w), int(lm[2].y*h)), 4, (255, 0, 0), -1)
                    cv2.circle(frame, (int(lm[5].x*w), int(lm[5].y*h)), 4, (0, 0, 255), -1)
                    cv2.circle(frame, (int(lm[7].x*w), int(lm[7].y*h)), 4, (255, 0, 0), -1)
                    cv2.circle(frame, (int(lm[8].x*w), int(lm[8].y*h)), 4, (0, 0, 255), -1)
                    # かかと〜足先（ライン）
                    cv2.line(frame, (int(lm[29].x*w), int(lm[29].y*h)), (int(lm[31].x*w), int(lm[31].y*h)), (255, 0, 0), 2)
                    cv2.line(frame, (int(lm[30].x*w), int(lm[30].y*h)), (int(lm[32].x*w), int(lm[32].y*h)), (0, 0, 255), 2)

                # FFmpegへ書き込み
                try:
                    process.stdin.write(frame.tobytes())
                except BrokenPipeError:
                    break

                if SHOW_WINDOW:
                    cv2.imshow('Batch Gait Analysis', frame)
                    if cv2.waitKey(1) & 0xFF == 27: # ESCで強制中断
                        print("ユーザーによる中断")
                        break

                processed_frame_count += 1
        finally:
            # 1ファイルの終了処理
            f_csv.close()
            cap.release()
            if process.stdin:
                process.stdin.close()
            process.wait()
            if SHOW_WINDOW:
                cv2.destroyAllWindows()
        print(f"完了: {csv_output}")

def main():
    if not os.path.exists(CONFIG_FILE):
        print(f"エラー: 設定ファイル {CONFIG_FILE} が見つかりません。")
        print("以下のようなJSONファイルを作成してください:")
        print('[{"input_video": "video.mp4", "start_sec": 5, "end_sec": 10}]')
        return

    print(f"{CONFIG_FILE} を読み込んでいます...")
    try:
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            batch_list = json.load(f)
    except json.JSONDecodeError:
        print("エラー: JSONファイルの形式が正しくありません。")
        return

    total_files = len(batch_list)
    print(f"合計 {total_files} 件のファイルを処理します。")

    for i, config in enumerate(batch_list):
        print(f"\n--- [{i+1}/{total_files}] ---")
        process_video(config)

    cv2.destroyAllWindows()
    print("\nすべてのバッチ処理が完了しました。")

if __name__ == '__main__':
    main()
