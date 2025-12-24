WSL + Docker + VS Code 再開手順（mediapipe311）

1) WSL を開く
Windows のターミナルから Ubuntu を開く。
wsl

2) プロジェクトへ移動して VS Code を起動
cd /home/yasun/projects/mediapipe311
code .

3) VS Code でコンテナにアタッチ
VS Code 左下「><」→ Attach to Running Container... → mediapipe311-dev

4) /workspace を開く
VS Code → File → Open Folder... → /workspace
ここに plot_gait.py / gait_analysis_batch.py が見える。

5) コンテナ内ターミナルで実行
cd /workspace
python gait_analysis_batch.py --batch-config batch_config.json

補足: コンテナが起動していない場合
WSL で以下を実行して起動する。
docker ps
docker start mediapipe311-dev

コンテナ操作（よく使うもの）
1) 起動中の一覧
docker ps

2) 停止中を含む一覧
docker ps -a

3) 特定コンテナを起動
docker start <container_name>
例: docker start mediapipe311-dev

4) 特定コンテナを停止
docker stop <container_name>
例: docker stop mediapipe311-dev

5) 特定コンテナを作成して起動（今回の構成）
docker run -d --name mediapipe311-dev \
  -v /home/yasun/projects/mediapipe311:/workspace \
  -v /mnt/c/Python/mmpose/mp4:/data/mp4 \
  -v /mnt/c/Python/mmpose/output:/output \
  -v mmpose311_pip-cache:/home/devuser/.cache/pip \
  mmpose311-dev sleep infinity

補足: Attach 後のターミナルはコンテナ内
コンテナ内では docker コマンドは使えない。
code . は WSL 側で実行する。
