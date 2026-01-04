# Snowboard AI

スノーボードのグラウンドトリックを解析し、改善点を提案する AI ツールです。
Web ブラウザ上で操作できる GUI モードと、コマンドラインツールの両方を提供しています。

## セットアップ

1. 仮想環境の作成と有効化

   **Windows (PowerShell):**

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

   **Windows (コマンドプロンプト):**

   ```cmd
   python -m venv venv
   venv\Scripts\activate.bat
   ```

   **Linux/Mac:**

   ```bash
   python -m venv venv
   source venv/Scripts/activate
   ```

2. 依存ライブラリのインストール
   ```bash
   pip install tensorflow tensorflow-hub opencv-python numpy matplotlib streamlit
   ```

## 使い方

### 1. Web アプリ（GUI）での利用

ブラウザ上で動画のアップロード、解析、結果のダウンロードが可能です。

```bash
streamlit run src/web_app.py
```

**利用可能な機能：**

1. **解析 (Analyze)**: 動画をアップロードして技の解析を行う
2. **教師データ追加 (Add Training Data)**: 動画をアップロードして教師データとして追加
   - 動画をアップロード
   - 特徴量を自動抽出
   - UI 上でラベル付け（技分類と成功判定）
   - ラベルを保存
3. **学習 (Train)**: 登録された教師データでモデルを再学習

### 2. コマンドラインでの利用

#### 推論（解析）

```bash
python main.py --input data/input/my_trick.mp4
```

**注意:** コマンドはプロジェクトのルートディレクトリ（`snowboard_ai`）から実行してください。

結果は `data/output/output.mp4` として保存され、コンソールにアドバイスが表示されます。

### 再学習

データが蓄積されたら、モデルを再学習できます。

#### 1. 動画から特徴量を抽出

```bash
python src/preprocessing/extract_and_save_features.py --input data/input/my_trick.mp4 --output-dir data/features

```

#### 2. ラベル付け

対話的にラベル付けを行う：

```bash
python src/training/create_labels.py --mode interactive
```

コマンドラインで直接ラベルを追加する：

```bash
python src/training/create_labels.py --mode add --feature-file video1_features.npy --trick-class 0 --success-flag 1
```

ラベル一覧を表示：

```bash
python src/training/create_labels.py --mode list
```

#### 3. モデル学習

```bash
python src/training/train_model.py --data-dir data/features --labels-file data/labels/labels.json --model-path models/trained/snowboard_lstm.h5 --epochs 50
```

**注意:** コマンドはプロジェクトのルートディレクトリ（`snowboard_ai`）から実行してください。

## プロジェクト構造

```
snowboard_ai/
├── main.py                 # コマンドライン実行用のメインスクリプト
├── data/
│   ├── input/              # 入力動画
│   ├── labels/             # 教師ラベル（JSON）
│   ├── features/           # 抽出された特徴量（numpy）
│   └── output/             # 解析結果動画
├── models/
│   └── trained/            # 学習済みモデル
├── src/
│   ├── web_app.py         # Streamlit Webアプリケーション
│   ├── pose/              # 姿勢推定モジュール
│   ├── preprocessing/     # 特徴量抽出モジュール
│   ├── analysis/          # 解析・推論モジュール
│   ├── training/          # モデル学習モジュール
│   └── visualize/         # 可視化モジュール
├── design.md              # 設計ドキュメント
└── README.md
```

## データ投入と教師データ作成の詳細手順

### ステップ 1: 動画データの準備と投入

#### 1.1 動画の取得と切り出し

**基本的な流れ：**

1. YouTube などのハウツー動画から必要な部分を切り出す
2. 切り出した動画を `data/input/` に配置する
3. 特徴量を抽出する（**動画そのものは学習に使わない**）

**重要なポイント：**

- **動画そのものは学習させません**
- MoveNet で抽出した**骨格・特徴量のみ**を学習対象とします
- これにより、YouTube 動画の著作権リスクを低減できます

#### 1.2 動画の切り出し方法

**方法 1: ffmpeg を使用（推奨）**

ffmpeg をインストールしていない場合：

**Windows でのインストール方法：**

**方法 A: 実行ファイルをダウンロード（推奨）**

1. [ffmpeg 公式サイト](https://ffmpeg.org/download.html) にアクセス
2. Windows 用のビルドをダウンロード（例：[BtbN/ffmpeg-builds](https://github.com/BtbN/ffmpeg-builds/releases)）
3. ダウンロードした ZIP を解凍（例：`C:\ffmpeg\`）
4. `C:\ffmpeg\bin\` を環境変数 PATH に追加
   - 「システムのプロパティ」→「環境変数」→「システム環境変数」の「Path」を編集
   - `C:\ffmpeg\bin` を追加
5. PowerShell を再起動して確認：
   ```powershell
   ffmpeg -version
   ```

**方法 B: Chocolatey を使用**

```powershell
choco install ffmpeg
```

**方法 C: winget を使用（Windows 10/11）**

```powershell
winget install ffmpeg
```

**Linux/Mac でのインストール方法：**

```bash
# Ubuntu/Debian
sudo apt update
sudo apt install ffmpeg

# Mac (Homebrew)
brew install ffmpeg
```

**動画の一部を切り出す：**

```bash
# 例：00:30から10秒間を切り出す
ffmpeg -i input_video.mp4 -ss 00:00:30 -t 00:00:10 -c copy data/input/nose_press_001.mp4

# 例：1分30秒から20秒間を切り出す
ffmpeg -i input_video.mp4 -ss 00:01:30 -t 00:00:20 -c copy data/input/tail_press_001.mp4
```

**パラメータ説明：**

- `-i`: 入力動画ファイル
- `-ss`: 開始時刻（時:分:秒）
- `-t`: 切り出す長さ（秒）
- `-c copy`: 再エンコードせずにコピー（高速）

**複数の部分を一度に切り出す：**

```bash
# 例：複数の技を一度に切り出す
ffmpeg -i tutorial.mp4 -ss 00:00:30 -t 00:00:10 -c copy data/input/nose_press_001.mp4
ffmpeg -i tutorial.mp4 -ss 00:01:20 -t 00:00:12 -c copy data/input/nose_press_002.mp4
ffmpeg -i tutorial.mp4 -ss 00:02:45 -t 00:00:15 -c copy data/input/tail_press_001.mp4
```

**方法 2: 提供スクリプトを使用**

プロジェクトに含まれているスクリプトを使用する場合：

**Windows PowerShell:**

```powershell
.\scripts\extract_video_segment.ps1 -InputVideo "tutorial.mp4" -StartTime "00:00:30" -Duration 10 -OutputName "nose_press_001.mp4"
```

**Linux/Mac:**

```bash
bash scripts/extract_video_segment.sh tutorial.mp4 00:00:30 10 nose_press_001.mp4
```

**方法 3: Python/OpenCV を使用（ffmpeg 不要）**

ffmpeg がインストールされていない場合、Python と OpenCV を使用して動画を切り出すことができます。

```bash
# 18分00秒から5秒間を切り出す例
python scripts/extract_video_segment_python.py tutorial.f399.mp4 00:18:00 5 nose_press_001.mp4

# または秒数で指定
python scripts/extract_video_segment_python.py tutorial.f399.mp4 1080 5 nose_press_001.mp4
```

**パラメータ説明：**

- 第 1 引数: 入力動画ファイル
- 第 2 引数: 開始時刻（`HH:MM:SS` 形式または秒数）
- 第 3 引数: 切り出す長さ（秒）
- 第 4 引数: 出力ファイル名

**注意：**

- OpenCV を使用するため、ffmpeg より処理が遅い場合があります
- 再エンコードが発生するため、画質が若干劣化する可能性があります

**方法 4: 動画編集ソフトを使用**

- Windows: 動画エディタ、Windows 標準の「フォト」アプリなど
- Mac: QuickTime Player、iMovie など
- オンライン: [Clideo](https://clideo.com/cut-video)、[Kapwing](https://www.kapwing.com/)など

**方法 5: YouTube 動画を直接ダウンロードして切り出す（yt-dlp 使用）**

yt-dlp は、YouTube やその他の動画サイトから動画をダウンロードできるコマンドラインツールです。

##### yt-dlp のインストール方法

**Windows:**

**方法 A: pip でインストール（推奨）**

```bash
pip install yt-dlp
```

**方法 B: 実行ファイルをダウンロード**

1. [yt-dlp のリリースページ](https://github.com/yt-dlp/yt-dlp/releases)から `yt-dlp.exe` をダウンロード
2. 適当なフォルダ（例：`C:\tools\`）に配置
3. 環境変数 PATH に追加するか、フルパスで実行

**方法 C: Chocolatey を使用**

```powershell
choco install yt-dlp
```

**Linux/Mac:**

```bash
# pip でインストール（推奨）
pip install yt-dlp

# または Homebrew（Mac）
brew install yt-dlp

# または apt（Ubuntu/Debian）
sudo apt install yt-dlp
```

##### yt-dlp の基本的な使い方

**動画をダウンロード：**

```bash
# 基本的なダウンロード（最高画質）
yt-dlp "https://www.youtube.com/watch?v=VIDEO_ID"

# 出力ファイル名を指定
yt-dlp "https://www.youtube.com/watch?v=VIDEO_ID" -o "tutorial.mp4"

# MP4形式で最高画質をダウンロード
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" "https://www.youtube.com/watch?v=VIDEO_ID" -o "tutorial.mp4"

# 短縮URLでも使用可能
yt-dlp "https://youtu.be/VIDEO_ID" -o "tutorial.mp4"
```

**よく使うオプション：**

```bash
# 特定の画質を指定
yt-dlp -f "best[height<=720]" "URL" -o "output.mp4"  # 720p以下

# 音声のみダウンロード
yt-dlp -f "bestaudio" -x --audio-format mp3 "URL" -o "audio.mp3"

# 動画情報を表示（ダウンロードしない）
yt-dlp --list-formats "URL"

# プレイリスト全体をダウンロード
yt-dlp "https://www.youtube.com/playlist?list=PLAYLIST_ID" -o "playlist_%(title)s.%(ext)s"
```

##### 学習データ作成での使用例

**方法 A: コマンドラインで直接実行**

```bash
# 1. YouTube動画をダウンロード
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" "https://www.youtube.com/watch?v=VIDEO_ID" -o tutorial.mp4

# 2. 必要な部分を切り出して data/input/ に配置
ffmpeg -i tutorial.mp4 -ss 00:00:30 -t 00:00:10 -c copy data/input/nose_press_001.mp4
ffmpeg -i tutorial.mp4 -ss 00:01:20 -t 00:00:12 -c copy data/input/nose_press_002.mp4
ffmpeg -i tutorial.mp4 -ss 00:02:45 -t 00:00:15 -c copy data/input/tail_press_001.mp4

# 3. 元の動画ファイルは削除してもOK（data/input/ に切り出し済み）
del tutorial.mp4
```

**方法 B: 提供スクリプトを使用**

```powershell
# Windows PowerShell: YouTube動画をダウンロード
.\scripts\download_youtube_video.ps1 -Url "https://www.youtube.com/watch?v=VIDEO_ID" -OutputName "tutorial.mp4"

# その後、動画を切り出す
.\scripts\extract_video_segment.ps1 -InputVideo "tutorial.mp4" -StartTime "00:00:30" -Duration 10 -OutputName "nose_press_001.mp4"
```

```bash
# Linux/Mac: YouTube動画をダウンロード
bash scripts/download_youtube_video.sh "https://www.youtube.com/watch?v=VIDEO_ID" tutorial.mp4

# その後、動画を切り出す
bash scripts/extract_video_segment.sh tutorial.mp4 00:00:30 10 nose_press_001.mp4
```

**注意事項：**

- YouTube の利用規約を確認し、個人利用の範囲内で使用してください
- ダウンロードした動画は学習用の特徴量抽出のみに使用し、動画そのものは学習に使用しません
- 著作権を尊重し、適切に使用してください

#### 1.3 動画ファイルの配置

切り出した動画ファイルを `data/input/` ディレクトリに配置します。

```bash
# Windows
copy C:\path\to\your\video.mp4 data\input\my_trick_001.mp4

# Linux/Mac
cp /path/to/your/video.mp4 data/input/my_trick_001.mp4
```

**ファイル命名の推奨：**

- `{技名}_{番号}.mp4` の形式を推奨
- 例：`nose_press_001.mp4`, `tail_press_001.mp4`, `butter_180_001.mp4`
- 成功例と失敗例を区別する場合：`nose_press_success_001.mp4`, `nose_press_fail_001.mp4`

**注意事項：**

- 対応形式：MP4（推奨）、AVI、MOV など OpenCV が読み込める形式
- 動画の内容：**1 つの動画に 1 つの技が含まれていることを推奨**
- 動画の長さ：3〜10 秒程度が目安（短すぎると特徴量が少ない、長すぎると不要な部分が含まれる）

#### 1.4 特徴量の抽出

動画から骨格情報を抽出し、特徴量として保存します。

**基本的な使い方：**

```bash
python src/preprocessing/extract_and_save_features.py --input data/input/my_trick_001.mp4 --output-dir data/features
```

**実行結果：**

- `data/features/my_trick_001_features.npy` が作成されます
- このファイルには、動画の各フレームから抽出された特徴量が保存されます

**複数の動画を一括処理する場合：**

**方法 1: PowerShell スクリプトを使用（推奨）**

```powershell
.\scripts\batch_extract_features.ps1
```

**方法 2: 手動でループ処理**

Windows PowerShell の場合：

```powershell
Get-ChildItem data\input\*.mp4 | ForEach-Object {
    python src/preprocessing/extract_and_save_features.py --input $_.FullName --output-dir data/features
}
```

Linux/Mac の場合：

```bash
bash scripts/batch_extract_features.sh
```

**オプション：**

- `--input`: 入力動画のパス（必須）
- `--output-dir`: 特徴量を保存するディレクトリ（デフォルト: `data/features`）

### ステップ 2: 教師データ（ラベル）の作成

抽出した特徴量に対して、技の種類と成功/失敗のラベルを付けます。

#### 2.1 ラベルの種類

**技分類（trick_class）：**

- `0`: nose_press（ノーズプレス）
- `1`: tail_press（テールプレス）
- `2`: butter_180（バター 180）

**技分類の定義場所と追加方法：**

技分類は以下のファイルで定義されています：

1. **`src/training/create_labels.py`** - `TRICK_CLASSES` 辞書（13-17 行目）
2. **`main.py`** - `trick_names` 辞書（91 行目）
3. **`src/web_app.py`** - `trick_names` 辞書（139 行目）
4. **`design.md`** - 設計ドキュメント（126-128 行目）

**新しい技を追加する手順：**

例：`nose_ollie`（ノーズオーリー）を追加する場合：

1. **`src/training/create_labels.py` を編集：**

   ```python
   TRICK_CLASSES = {
       0: "nose_press",
       1: "tail_press",
       2: "butter_180",
       3: "nose_ollie"  # 追加
   }
   ```

2. **`main.py` を編集：**

   ```python
   trick_names = {0: "Nose Press", 1: "Tail Press", 2: "Butter 180", 3: "Nose Ollie"}  # 追加
   ```

3. **`src/web_app.py` を編集：**

   ```python
   trick_names = {0: "Nose Press", 1: "Tail Press", 2: "Butter 180", 3: "Nose Ollie"}  # 追加
   ```

4. **`design.md` を更新：**

   ```
   0 = nose_press
   1 = tail_press
   2 = butter_180
   3 = nose_ollie  # 追加
   ```

5. **`src/training/train_model.py` のデフォルト値を更新：**

   ```python
   # 学習時に num_classes を指定（例：4クラスの場合）
   python src/training/train_model.py --epochs 50
   # または train_model.py 内のデフォルト値を変更
   trainer = SnowboardTrainer(input_shape, num_classes=4)  # 3 → 4 に変更
   ```

6. **既存のモデルを再学習：**
   - 新しい技のデータを追加してラベル付け
   - モデルを再学習（`num_classes` を新しいクラス数に変更）

**注意事項：**

- 技を追加した後は、既存のモデルを再学習する必要があります
- 既存のラベルデータ（`data/labels/labels.json`）はそのまま使用できます
- 新しい技のデータを追加してから再学習してください

**成功判定（success_flag）：**

- `0`: NG（失敗・未完成）
- `1`: OK（成功・完成）

#### 2.2 Web アプリでラベル付け（最も簡単・推奨）

Web アプリの UI 上で動画をアップロードし、ラベル付けができます。

1. Web アプリを起動：

   ```bash
   streamlit run src/web_app.py
   ```

2. サイドバーで「教師データ追加 (Add Training Data)」を選択

3. 動画をアップロード

4. 「特徴量を抽出」ボタンをクリック

5. 技分類と成功判定を選択

6. 「ラベルを保存」ボタンをクリック

これで教師データが追加されます！

#### 2.3 対話的ラベル付け（コマンドライン）

コマンドラインで対話形式でラベルを付けられます。

```bash
python src/training/create_labels.py --mode interactive
```

**操作手順：**

1. 特徴量ファイルの一覧が表示されます

   ```
   === Feature Files ===
   1. [ ] my_trick_001_features.npy
   2. [ ] my_trick_002_features.npy
   3. [✓] my_trick_003_features.npy  ← 既にラベルが付いているものは✓表示
   ```

2. ラベルを付けたいファイルの番号を入力

   ```
   Enter feature file number to label (or 'q' to quit):
   > 1
   ```

3. 技分類を入力（0-2）

   ```
   Trick class (0-2): 0
   ```

4. 成功判定を入力（0: NG, 1: OK）

   ```
   Success flag (0: NG, 1: OK): 1
   ```

5. ラベルが保存されます

   ```
   Label added: my_trick_001_features.npy -> Trick: nose_press, Success: OK
   ```

6. `q` を入力して終了

#### 2.4 コマンドラインで直接ラベルを追加

既にラベルが分かっている場合は、コマンドラインで直接追加できます。

```bash
python src/training/create_labels.py --mode add --feature-file my_trick_001_features.npy --trick-class 0 --success-flag 1
```

**例：複数のラベルを一括追加**

```bash
# Nose Press (成功)
python src/training/create_labels.py --mode add --feature-file nose_press_001_features.npy --trick-class 0 --success-flag 1

# Nose Press (失敗)
python src/training/create_labels.py --mode add --feature-file nose_press_002_features.npy --trick-class 0 --success-flag 0

# Tail Press (成功)
python src/training/create_labels.py --mode add --feature-file tail_press_001_features.npy --trick-class 1 --success-flag 1

# Butter 180 (成功)
python src/training/create_labels.py --mode add --feature-file butter_180_001_features.npy --trick-class 2 --success-flag 1
```

#### 2.5 ラベルの確認と管理

**ラベル一覧を表示：**

```bash
python src/training/create_labels.py --mode list
```

**出力例：**

```
=== Registered Labels ===
my_trick_001_features.npy: nose_press (OK)
my_trick_002_features.npy: tail_press (NG)
my_trick_003_features.npy: butter_180 (OK)
```

**ラベルを削除：**

```bash
python src/training/create_labels.py --mode remove --feature-file my_trick_001_features.npy
```

#### 2.6 ラベルファイルの直接編集

ラベルは `data/labels/labels.json` に保存されます。直接編集することも可能です。

```json
{
  "my_trick_001_features.npy": {
    "trick_class": 0,
    "success_flag": 1
  },
  "my_trick_002_features.npy": {
    "trick_class": 1,
    "success_flag": 0
  }
}
```

### ステップ 3: モデルの学習

ラベルが付いたデータを使ってモデルを学習します。

**基本的な使い方：**

```bash
python src/training/train_model.py --data-dir data/features --labels-file data/labels/labels.json --model-path models/trained/snowboard_lstm.h5 --epochs 50
```

**オプション：**

- `--data-dir`: 特徴量ファイルが保存されているディレクトリ（デフォルト: `data/features`）
- `--labels-file`: ラベルファイルのパス（デフォルト: `data/labels/labels.json`）
- `--model-path`: 学習済みモデルの保存先（デフォルト: `models/trained/snowboard_lstm.h5`）
- `--epochs`: 学習エポック数（デフォルト: 50）
- `--batch-size`: バッチサイズ（デフォルト: 32）
- `--target-timesteps`: モデル入力用のタイムステップ数（デフォルト: 60）

**学習の進行状況：**

学習中は以下のような情報が表示されます：

```
Loading data from: data/features
Loaded 10 labels from data/labels/labels.json
Found 10 feature files
Loaded data shape: X=(10, 60, 12), y_class=(10,), y_success=(10,)

=== Training Configuration ===
Data shape: (10, 60, 12)
Epochs: 50
Batch size: 32
Model will be saved to: models/trained/snowboard_lstm.h5

Epoch 1/50
...
```

## 再学習フロー（まとめ）

1. **動画を `data/input/` に配置**

   ```bash
   # 動画ファイルをコピー
   copy your_video.mp4 data\input\
   ```

2. **特徴量抽出**

   ```bash
   python src/preprocessing/extract_and_save_features.py --input data/input/your_video.mp4
   ```

3. **ラベル付け**

   ```bash
   python src/training/create_labels.py --mode interactive
   ```

4. **モデル学習**
   ```bash
   python src/training/train_model.py --epochs 50
   ```

## よくある質問（FAQ）

### Q: どのくらいのデータが必要ですか？

A: 最低でも各技につき 10 サンプル以上あると良いでしょう。より多くのデータがあるほど、モデルの精度が向上します。

### Q: ラベルを間違えて付けました。修正できますか？

A: はい。対話モードで同じファイル番号を選択して再度ラベルを付け直すか、`--mode remove` で削除してから再追加してください。

### Q: 特徴量ファイルは削除しても大丈夫ですか？

A: 特徴量ファイルを削除した場合、対応するラベルも削除する必要があります。動画ファイルは `data/input/` に残しておけば、いつでも再抽出できます。

### Q: 学習に時間がかかります。どうすればいいですか？

A: `--epochs` の値を減らすか、`--batch-size` を増やすことで学習時間を短縮できます。ただし、精度に影響する可能性があります。

### Q: モデルが読み込めません

A: モデルファイル（`.h5`）が `models/trained/` に存在するか確認してください。存在しない場合は、まず学習を実行してください。

### Q: 教師データを増やさずに精度を上げる方法はありますか？

A: はい、以下の方法があります：

1. **データ拡張**: 既存の特徴量データを変形して増やす

   ```bash
   python src/training/train_model_improved.py --epochs 100
   ```

2. **改善されたモデル**: より深いネットワークと正則化を使用

   - `train_model_improved.py` を使用
   - データ拡張、BatchNormalization、学習率スケジューリングを含む

3. **特徴量の改善**: より多くの特徴量を抽出
   - `extract_features_improved.py` を使用

詳細は「精度向上のための手法」セクションを参照してください。

## 実用例：初めての学習データ作成

以下は、実際に学習データを作成する手順の例です。

### 例：YouTube 動画から学習データを作成する

```bash
# 1. YouTube動画をダウンロード（yt-dlp使用）
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]" "https://www.youtube.com/watch?v=VIDEO_ID" -o tutorial.mp4

# 2. 必要な部分を切り出して data/input/ に配置
ffmpeg -i tutorial.mp4 -ss 00:00:30 -t 00:00:10 -c copy data/input/nose_press_001.mp4
ffmpeg -i tutorial.mp4 -ss 00:01:20 -t 00:00:12 -c copy data/input/nose_press_002.mp4
ffmpeg -i tutorial.mp4 -ss 00:02:45 -t 00:00:15 -c copy data/input/tail_press_001.mp4

# 3. 特徴量を抽出（動画から骨格データを抽出）
.\scripts\batch_extract_features.ps1

# 4. ラベル付け
python src/training/create_labels.py --mode interactive
# → 各ファイルに対して技分類と成功判定を入力

# 5. モデルを学習
python src/training/train_model.py --epochs 50
```

**ポイント：**

- 動画ファイル自体は `data/input/` に保存されますが、学習には使用されません
- 学習に使用されるのは、MoveNet で抽出された骨格・特徴量データのみです
- これにより、YouTube 動画の著作権リスクを低減できます

### 例：Nose Press の学習データを 3 つ追加する

```bash
# 1. 動画ファイルを配置
copy C:\Videos\nose_press_001.mp4 data\input\
copy C:\Videos\nose_press_002.mp4 data\input\
copy C:\Videos\nose_press_003.mp4 data\input\

# 2. 特徴量を一括抽出
.\scripts\batch_extract_features.ps1

# 3. ラベル付け（対話モード）
python src/training/create_labels.py --mode interactive
# → 各ファイルに対して以下を入力：
#    nose_press_001_features.npy: trick_class=0, success_flag=1 (成功)
#    nose_press_002_features.npy: trick_class=0, success_flag=1 (成功)
#    nose_press_003_features.npy: trick_class=0, success_flag=0 (失敗)

# 4. ラベルを確認
python src/training/create_labels.py --mode list

# 5. モデルを学習（データが少ない場合はエポック数を減らす）
python src/training/train_model.py --epochs 20
```

### 例：複数の技を一度に学習する

```bash
# 1. すべての動画を配置
copy C:\Videos\*.mp4 data\input\

# 2. 特徴量を一括抽出
.\scripts\batch_extract_features.ps1

# 3. ラベルを一括追加（コマンドライン）
python src/training/create_labels.py --mode add --feature-file nose_press_001_features.npy --trick-class 0 --success-flag 1
python src/training/create_labels.py --mode add --feature-file nose_press_002_features.npy --trick-class 0 --success-flag 0
python src/training/create_labels.py --mode add --feature-file tail_press_001_features.npy --trick-class 1 --success-flag 1
python src/training/create_labels.py --mode add --feature-file butter_180_001_features.npy --trick-class 2 --success-flag 1

# 4. モデルを学習
python src/training/train_model.py --epochs 50
```

### 例：既存のモデルを改善する（追加学習）

```bash
# 1. 新しい動画を追加
copy C:\Videos\new_trick.mp4 data\input\

# 2. 特徴量を抽出
python src/preprocessing/extract_and_save_features.py --input data/input/new_trick.mp4

# 3. ラベルを追加
python src/training/create_labels.py --mode add --feature-file new_trick_features.npy --trick-class 0 --success-flag 1

# 4. 既存のデータと合わせて再学習
python src/training/train_model.py --epochs 50
```

## データ管理のベストプラクティス

1. **ファイル命名規則**

   - 動画ファイル：`{技名}_{番号}.mp4`（例：`nose_press_001.mp4`）
   - 特徴量ファイル：自動生成される（`{動画名}_features.npy`）

2. **データの整理**

   - 定期的に `data/features/` と `data/labels/labels.json` を確認
   - 不要な特徴量ファイルは削除（対応するラベルも削除）

3. **バックアップ**

   - `data/labels/labels.json` は定期的にバックアップを取る
   - 動画ファイルは `data/input/` に残しておく（再抽出可能）

4. **データの品質**
   - 各技につき、成功例と失敗例の両方を用意する
   - 動画は 1 つの技が明確に含まれているものを使用

## 精度向上のための手法（教師データを増やさずに）

既存の教師データを増やさずにモデルの精度を向上させる方法を提供しています。

### 1. データ拡張（Data Augmentation）

既存の特徴量データを変形してデータを増やします。

**手法：**

- **ノイズ追加**: ガウシアンノイズを追加してロバスト性を向上
- **時間シフト**: 時間軸方向にフレームをずらす
- **スケーリング**: 特徴量を拡大・縮小
- **時間ワープ**: 時間の伸縮をシミュレート

**使用方法：**

```bash
# 改善されたモデルで学習（データ拡張自動適用）
python src/training/train_model_improved.py --epochs 100
```

### 2. 改善されたモデルアーキテクチャ

より深いネットワークと正則化技術を使用：

- **より深い LSTM 層**: 128 → 64 → 32 ユニット
- **BatchNormalization**: 学習の安定化
- **Dropout の調整**: 過学習の防止
- **学習率スケジューリング**: 自動的な学習率調整
- **Early Stopping**: 過学習の防止
- **Top-K Accuracy**: より詳細な評価指標

**使用方法：**

```bash
# 改善されたモデルで学習
python src/training/train_model_improved.py --epochs 100 --batch-size 32

# データ拡張を無効にする場合
python src/training/train_model_improved.py --epochs 100 --no-augmentation
```

### 3. 改善された特徴量抽出

より多くの特徴量を抽出して精度を向上：

**追加特徴量：**

- 肩と腰の距離（体の開き具合）
- 体の傾き
- 左右の膝の角度差（バランス）
- 重心位置（重み付き平均）
- 体の幅（肩幅と腰幅）
- 頭の位置
- 左右のバランス
- 加速度（2 次微分）

**使用方法：**

`extract_features_improved.py` を使用するには、`extract_and_save_features.py` を修正して `ImprovedFeatureExtractor` を使用するように変更してください。

### 4. アンサンブル学習（将来実装）

複数のモデルを組み合わせて精度を向上させる方法です。

### 実装例：精度向上のワークフロー

```bash
# 1. 既存のデータで改善されたモデルを学習
python src/training/train_model_improved.py --epochs 100

# 2. 結果を確認
# models/trained/best_model.h5 が自動保存されます

# 3. 必要に応じてエポック数を調整
python src/training/train_model_improved.py --epochs 150 --batch-size 16
```

**期待される効果：**

- データ拡張により、実質的なデータ量が 2-3 倍に
- より深いネットワークにより、複雑なパターンを学習
- 正則化により、過学習を抑制
- 学習率スケジューリングにより、より良い収束

**注意事項：**

- データ拡張を使用すると学習時間が長くなります
- 改善されたモデルは、より多くのメモリを使用します
- 少量のデータでも効果がありますが、ある程度のデータ（各技 10 サンプル以上）があるとより効果的です

詳細は `design.md` を参照してください。
