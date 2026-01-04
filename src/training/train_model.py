import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import os
import json
from pathlib import Path

class SnowboardTrainer:
    def __init__(self, input_shape, num_classes):
        self.input_shape = input_shape # (time_steps, features)
        self.num_classes = num_classes
        self.model = self._build_model()

    def _build_model(self):
        """LSTMモデルの構築"""
        input_layer = layers.Input(shape=self.input_shape)
        
        # LSTM層
        x = layers.LSTM(64, return_sequences=True)(input_layer)
        x = layers.Dropout(0.2)(x)
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.2)(x)
        
        # 技分類用ヘッド
        class_output = layers.Dense(self.num_classes, activation='softmax', name='trick_class')(x)
        
        # 成功判定用ヘッド
        success_output = layers.Dense(1, activation='sigmoid', name='success_flag')(x)
        
        model = models.Model(inputs=input_layer, outputs=[class_output, success_output])
        
        model.compile(
            optimizer='adam',
            loss={
                'trick_class': 'sparse_categorical_crossentropy',
                'success_flag': 'binary_crossentropy'
            },
            loss_weights={'trick_class': 1.0, 'success_flag': 0.5},
            metrics={'trick_class': 'accuracy', 'success_flag': 'accuracy'}
        )
        return model

    def train(self, X_train, y_class_train, y_success_train, epochs=50, batch_size=32):
        history = self.model.fit(
            X_train,
            {'trick_class': y_class_train, 'success_flag': y_success_train},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.2
        )
        return history

    def save_model(self, path):
        self.model.save(path)

def load_data(data_dir, labels_file=None, target_timesteps=60):
    """
    データの読み込み（numpyファイルとラベルファイルから）
    
    Args:
        data_dir: 特徴量ファイルが保存されているディレクトリ
        labels_file: ラベルファイルのパス（Noneの場合はdata_dirの親ディレクトリのlabels/labels.jsonを探す）
        target_timesteps: モデル入力用のタイムステップ数（リサンプリング用）
    
    Returns:
        X: (samples, time_steps, features) 特徴量データ
        y_class: (samples,) 技分類ラベル
        y_success: (samples,) 成功判定ラベル
    """
    print("Loading data from:", data_dir)
    
    # ラベルファイルのパス決定
    if labels_file is None:
        # data_dirの親ディレクトリのlabels/labels.jsonを探す
        data_path = Path(data_dir)
        labels_file = data_path.parent / "labels" / "labels.json"
    
    # ラベル読み込み
    labels = {}
    if os.path.exists(labels_file):
        with open(labels_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        print(f"Loaded {len(labels)} labels from {labels_file}")
    else:
        print(f"Warning: Labels file not found: {labels_file}")
        print("Using dummy labels (random)")
    
    # 特徴量ファイル一覧取得
    feature_files = [f for f in os.listdir(data_dir) if f.endswith('_features.npy')]
    
    if not feature_files:
        print("Warning: No feature files found. Using dummy data.")
        # ダミーデータ生成
        num_samples = 100
        X = np.random.rand(num_samples, target_timesteps, 12)
        y_class = np.random.randint(0, 3, num_samples)
        y_success = np.random.randint(0, 2, num_samples)
        return X, y_class, y_success
    
    print(f"Found {len(feature_files)} feature files")
    
    # データ読み込み
    X_list = []
    y_class_list = []
    y_success_list = []
    
    for feature_file in feature_files:
        feature_path = os.path.join(data_dir, feature_file)
        
        # 特徴量読み込み
        features = np.load(feature_path)  # (frames, num_features)
        
        # タイムステップ数にリサンプリング
        current_timesteps = features.shape[0]
        if current_timesteps > 0:
            if current_timesteps >= target_timesteps:
                # ダウンサンプリング
                indices = np.linspace(0, current_timesteps - 1, target_timesteps).astype(int)
                resampled_features = features[indices]
            else:
                # アップサンプリング（最後のフレームを繰り返し）
                padding = target_timesteps - current_timesteps
                resampled_features = np.vstack([features, np.tile(features[-1:], (padding, 1))])
        else:
            print(f"Warning: Empty features in {feature_file}, skipping")
            continue
        
        X_list.append(resampled_features)
        
        # ラベル取得（複数ラベル対応）
        if feature_file in labels:
            label_data = labels[feature_file]
            
            # 複数ラベル対応：リスト形式の場合、最初のラベルを使用
            if isinstance(label_data, list):
                if len(label_data) > 0:
                    # 複数ラベルがある場合、最初のラベルを使用（またはランダムに選択）
                    label = label_data[0]  # 最初のラベルを使用
                    y_class_list.append(label["trick_class"])
                    y_success_list.append(label["success_flag"])
                else:
                    print(f"Warning: Empty label list for {feature_file}, using random label")
                    y_class_list.append(np.random.randint(0, 3))
                    y_success_list.append(np.random.randint(0, 2))
            else:
                # 後方互換性：単一ラベル形式
                y_class_list.append(label_data["trick_class"])
                y_success_list.append(label_data["success_flag"])
        else:
            # ラベルがない場合はランダム（警告表示）
            print(f"Warning: No label for {feature_file}, using random label")
            y_class_list.append(np.random.randint(0, 3))
            y_success_list.append(np.random.randint(0, 2))
    
    # numpy配列に変換
    X = np.array(X_list)  # (samples, time_steps, features)
    y_class = np.array(y_class_list)  # (samples,)
    y_success = np.array(y_success_list)  # (samples,)
    
    print(f"Loaded data shape: X={X.shape}, y_class={y_class.shape}, y_success={y_success.shape}")
    
    return X, y_class, y_success

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train snowboard trick classification model')
    parser.add_argument('--data-dir', type=str, default='data/features',
                       help='Directory containing feature files')
    parser.add_argument('--labels-file', type=str, default='data/labels/labels.json',
                       help='Path to labels JSON file')
    parser.add_argument('--model-path', type=str, default='models/trained/snowboard_lstm.h5',
                       help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--target-timesteps', type=int, default=60,
                       help='Target timesteps for model input')
    
    args = parser.parse_args()
    
    # 絶対パスに変換
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(args.data_dir)
    if not os.path.isabs(args.labels_file):
        args.labels_file = os.path.abspath(args.labels_file)
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.abspath(args.model_path)
    
    # データロード
    X, y_c, y_s = load_data(args.data_dir, args.labels_file, args.target_timesteps)
    
    if len(X) == 0:
        print("Error: No data loaded. Please extract features and create labels first.")
        exit(1)
    
    # モデル初期化
    input_shape = (X.shape[1], X.shape[2])
    trainer = SnowboardTrainer(input_shape, num_classes=3)
    
    print(f"\n=== Training Configuration ===")
    print(f"Data shape: {X.shape}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Model will be saved to: {args.model_path}")
    print()
    
    # 学習
    trainer.train(X, y_c, y_s, epochs=args.epochs, batch_size=args.batch_size)
    
    # 保存
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    trainer.save_model(args.model_path)
    print(f"\nModel saved to {args.model_path}")