"""
改善された学習モデル
- データ拡張
- より深いネットワーク
- 正則化の改善
- 学習率スケジューリング
"""
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import numpy as np
import os
import json
from pathlib import Path

from preprocessing.data_augmentation import DataAugmentation

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
        num_samples = 100
        X = np.random.rand(num_samples, target_timesteps, 42) # Assuming 14 base features * 3 (val, vel, acc)
        y_class = np.random.randint(0, 3, num_samples)
        y_success = np.random.randint(0, 2, num_samples)
        return X, y_class, y_success
    
    print(f"Found {len(feature_files)} feature files")
    
    X_list, y_class_list, y_success_list = [], [], []
    
    for feature_file in feature_files:
        feature_path = os.path.join(data_dir, feature_file)
        features = np.load(feature_path)
        
        current_timesteps = features.shape[0]
        if current_timesteps > 0:
            if current_timesteps >= target_timesteps:
                indices = np.linspace(0, current_timesteps - 1, target_timesteps).astype(int)
                resampled_features = features[indices]
            else:
                padding = target_timesteps - current_timesteps
                resampled_features = np.vstack([features, np.tile(features[-1:], (padding, 1))])
        else:
            print(f"Warning: Empty features in {feature_file}, skipping")
            continue
        
        X_list.append(resampled_features)
        
        if feature_file in labels:
            label_data = labels[feature_file]
            if isinstance(label_data, list):
                if len(label_data) > 0:
                    label = label_data[0]
                    y_class_list.append(label["trick_class"])
                    y_success_list.append(label["success_flag"])
                else:
                    y_class_list.append(np.random.randint(0, 3))
                    y_success_list.append(np.random.randint(0, 2))
            else:
                y_class_list.append(label_data["trick_class"])
                y_success_list.append(label_data["success_flag"])
        else:
            print(f"Warning: No label for {feature_file}, using random label")
            y_class_list.append(np.random.randint(0, 3))
            y_success_list.append(np.random.randint(0, 2))
    
    X = np.array(X_list)
    y_class = np.array(y_class_list)
    y_success = np.array(y_success_list)
    
    print(f"Loaded data shape: X={X.shape}, y_class={y_class.shape}, y_success={y_success.shape}")
    
    return X, y_class, y_success


class SnowboardTrainer: # Renamed from ImprovedSnowboardTrainer
    def __init__(self, input_shape, num_classes, use_augmentation=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_augmentation = use_augmentation
        self.model = self._build_model()
        self.augmenter = DataAugmentation() if use_augmentation else None
    
    def _build_model(self):
        input_layer = layers.Input(shape=self.input_shape)
        x = layers.BatchNormalization()(input_layer)
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        class_output = layers.Dense(self.num_classes, activation='softmax', name='trick_class')(x)
        success_output = layers.Dense(1, activation='sigmoid', name='success_flag')(x)
        
        model = models.Model(inputs=input_layer, outputs=[class_output, success_output])
        
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate, decay_steps=100, decay_rate=0.96, staircase=True)
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss={'trick_class': 'sparse_categorical_crossentropy', 'success_flag': 'binary_crossentropy'},
            loss_weights={'trick_class': 1.0, 'success_flag': 0.5},
            metrics={'trick_class': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')], 'success_flag': 'accuracy'}
        )
        return model
    
    def train(self, X_train, y_class_train, y_success_train, epochs=100, batch_size=32, 
              validation_split=0.2, use_early_stopping=True):
        if self.use_augmentation:
            print("Applying data augmentation...")
            X_train, y_class_train, y_success_train = self.augmenter.augment_batch(
                X_train, y_class_train, y_success_train, augmentation_factor=2)
            print(f"Augmented data shape: {X_train.shape}")
        
        callback_list = []
        if use_early_stopping:
            callback_list.append(callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=1))
        
        callback_list.append(callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=1))
        callback_list.append(callbacks.ModelCheckpoint('models/trained/best_model.h5', monitor='val_loss', save_best_only=True, verbose=1))
        
        history = self.model.fit(
            X_train,
            {'trick_class': y_class_train, 'success_flag': y_success_train},
            epochs=epochs, batch_size=batch_size, validation_split=validation_split,
            callbacks=callback_list, verbose=1)
        
        return history
    
    def save_model(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved snowboard trick classification model')
    parser.add_argument('--data-dir', type=str, default='data/features', help='Directory containing feature files')
    parser.add_argument('--labels-file', type=str, default='data/labels/labels.json', help='Path to labels JSON file')
    parser.add_argument('--model-path', type=str, default='models/trained/snowboard_lstm.h5', help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--target-timesteps', type=int, default=60, help='Target timesteps for model input')
    parser.add_argument('--no-augmentation', action='store_true', help='Disable data augmentation')
    
    args = parser.parse_args()
    
    if not os.path.isabs(args.data_dir):
        args.data_dir = os.path.abspath(args.data_dir)
    if not os.path.isabs(args.labels_file):
        args.labels_file = os.path.abspath(args.labels_file)
    if not os.path.isabs(args.model_path):
        args.model_path = os.path.abspath(args.model_path)
    
    X, y_c, y_s = load_data(args.data_dir, args.labels_file, args.target_timesteps)
    
    if len(X) == 0:
        print("Error: No data loaded. Please extract features and create labels first.")
        exit(1)
    
    num_classes = len(np.unique(y_c))
    if num_classes == 0:
        print("Error: No classes found in the data. Please check labels.")
        exit(1)

    input_shape = (X.shape[1], X.shape[2])
    trainer = SnowboardTrainer(
        input_shape,
        num_classes=num_classes,
        use_augmentation=not args.no_augmentation)
    
    print(f"\n=== Training Configuration ===")
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data augmentation: {not args.no_augmentation}")
    print(f"Model will be saved to: {args.model_path}")
    print()
    
    trainer.train(X, y_c, y_s, epochs=args.epochs, batch_size=args.batch_size)
    
    trainer.save_model(args.model_path)
    print(f"\nModel saved to {args.model_path}")
