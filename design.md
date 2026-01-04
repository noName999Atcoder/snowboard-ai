# Snowboard AI（個人用・再学習あり）設計ドキュメント

## 1. プロジェクト概要

本プロジェクトは、**スノーボード（グラウンドトリック）の動作を動画から解析し、
再学習可能なAIモデルによって技の認識・評価・改善点提示を行う個人用ツール**である。

- 利用目的：個人利用（非商用）
- 再学習：あり
- 入力：MP4動画
- 出力：
  - 技の分類結果
  - 成功 / 未完成判定
  - 数値評価（角度・タイミング）
  - 改善ポイント（テキスト）

---

## 2. 技術スタック

### 言語・環境
- Python 3.10
- 仮想環境：venv
- OS：Windows（Git Bash）

### ライブラリ
- TensorFlow 2.13.0
- TensorFlow Hub
- MoveNet（SinglePose Lightning）
- OpenCV
- NumPy
- Matplotlib

---

## 3. 基本方針（重要）

### 3.1 再学習の考え方

- **動画そのものは学習させない**
- MoveNetで抽出した**骨格・特徴量のみを学習対象**とする

メリット：
- 学習が軽量
- 少量データでも成立
- 再学習が簡単
- YouTube動画の著作権リスクを低減

---

## 4. 全体アーキテクチャ

```
動画（MP4）
 ↓
OpenCV（フレーム取得）
 ↓
MoveNet（17 keypoints）
 ↓
骨格時系列データ
 ↓
特徴量変換
 ↓
MLモデル（再学習対象）
 ↓
技分類・成功判定
 ↓
ルール or LLMで改善点生成
```

---

## 5. ディレクトリ構成

```
snowboard_ai/
├── data/
│   ├── input/          # 入力動画
│   ├── labels/         # 教師ラベル
│   ├── features/       # 特徴量（numpy）
│   └── output/         # 出力結果
├── models/
│   └── trained/        # 学習済みモデル
├── src/
│   ├── pose/
│   │   └── movenet.py
│   ├── preprocessing/
│   │   └── extract_features.py
│   ├── training/
│   │   └── train_model.py
│   ├── analysis/
│   │   ├── inference.py
│   │   └── rules.py
│   ├── visualize/
│   │   └── draw_pose.py
│   └── main.py
├── design.md
└── README.md
```

---

## 6. 学習データ設計

### 6.1 入力データ（X）

MoveNet出力（17点）から以下を抽出する：

- 肩ライン角度
- 腰ライン角度
- 肩と腰のねじれ量
- 膝の曲げ量
- 回転角速度
- 荷重保持時間

```
X.shape = (samples, time_steps, features)
```

---

### 6.2 教師データ（y）

#### 技分類ラベル
```
0 = nose_press
1 = tail_press
2 = butter_180
```

#### 成功判定
```
0 = NG
1 = OK
```

#### フォームタイプ（任意）
```
0 = upper_body_first
1 = weight不足
2 = 回転不足
```

---

## 7. 学習モデル構成

### モデル候補
- LSTM
- Temporal CNN
- Simple Transformer（将来）

### 出力
- 技分類（softmax）
- 成功判定（sigmoid）

---

## 8. 再学習フロー

```
動画追加
 ↓
骨格抽出
 ↓
特徴量保存
 ↓
ラベル付け
 ↓
train_model.py 実行
 ↓
モデル更新
```

---

## 9. 改善点生成方針

- モデル出力は「判定」まで
- 改善コメントは**ルールベース**で生成

例：
- 肩回転が遅い →「回転の入りが遅れています」
- ノーズ荷重が短い →「ノーズに乗る時間が短いです」

---

## 10. 開発ステップ

### STEP 1（完了）
- 環境構築
- MoveNet動作確認
- GitHub接続

### STEP 2
- 特徴量抽出パイプライン作成

### STEP 3
- 学習コード作成（LSTM）

### STEP 4
- 推論 + 評価表示

---

## 11. 最終ゴール

- 自分の滑りを定量評価できる
- 再学習で精度が上がる
- 上達の指針として使える個人AI
