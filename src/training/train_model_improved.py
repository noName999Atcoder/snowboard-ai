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
import sys

# データ拡張モジュールをインポート
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from preprocessing.data_augmentation import DataAugmentation
from training.train_model import load_data


class ImprovedSnowboardTrainer:
    def __init__(self, input_shape, num_classes, use_augmentation=True):
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.use_augmentation = use_augmentation
        self.model = self._build_model()
        self.augmenter = DataAugmentation() if use_augmentation else None
    
    def _build_model(self):
        """改善されたLSTMモデルの構築"""
        input_layer = layers.Input(shape=self.input_shape)
        
        # BatchNormalizationを追加
        x = layers.BatchNormalization()(input_layer)
        
        # より深いLSTM層
        x = layers.LSTM(128, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.LSTM(64, return_sequences=True)(x)
        x = layers.Dropout(0.3)(x)
        x = layers.BatchNormalization()(x)
        
        x = layers.LSTM(32)(x)
        x = layers.Dropout(0.2)(x)
        
        # 追加のDense層
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # 技分類用ヘッド
        class_output = layers.Dense(self.num_classes, activation='softmax', name='trick_class')(x)
        
        # 成功判定用ヘッド
        success_output = layers.Dense(1, activation='sigmoid', name='success_flag')(x)
        
        model = models.Model(inputs=input_layer, outputs=[class_output, success_output])
        
        # 学習率スケジューリング付きのオプティマイザー
        initial_learning_rate = 0.001
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate,
            decay_steps=100,
            decay_rate=0.96,
            staircase=True
        )
        
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(
            optimizer=optimizer,
            loss={
                'trick_class': 'sparse_categorical_crossentropy',
                'success_flag': 'binary_crossentropy'
            },
            loss_weights={'trick_class': 1.0, 'success_flag': 0.5},
            metrics={
                'trick_class': ['accuracy', tf.keras.metrics.TopKCategoricalAccuracy(k=2, name='top2_accuracy')],
                'success_flag': 'accuracy'
            }
        )
        return model
    
    def train(self, X_train, y_class_train, y_success_train, epochs=100, batch_size=32, 
              validation_split=0.2, use_early_stopping=True):
        """
        学習実行（データ拡張とコールバックを含む）
        """
        # データ拡張を適用
        if self.use_augmentation:
            print("Applying data augmentation...")
            X_train, y_class_train, y_success_train = self.augmenter.augment_batch(
                X_train, y_class_train, y_success_train, augmentation_factor=2
            )
            print(f"Augmented data shape: {X_train.shape}")
        
        # コールバック設定
        callback_list = []
        
        if use_early_stopping:
            early_stopping = callbacks.EarlyStopping(
                monitor='val_loss',
                patience=15,
                restore_best_weights=True,
                verbose=1
            )
            callback_list.append(early_stopping)
        
        # 学習率削減
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
        callback_list.append(reduce_lr)
        
        # モデルチェックポイント
        checkpoint = callbacks.ModelCheckpoint(
            'models/trained/best_model.h5',
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
        callback_list.append(checkpoint)
        
        # 学習実行
        history = self.model.fit(
            X_train,
            {'trick_class': y_class_train, 'success_flag': y_success_train},
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            callbacks=callback_list,
            verbose=1
        )
        
        return history
    
    def save_model(self, path):
        """モデルを保存"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train improved snowboard trick classification model')
    parser.add_argument('--data-dir', type=str, default='data/features',
                       help='Directory containing feature files')
    parser.add_argument('--labels-file', type=str, default='data/labels/labels.json',
                       help='Path to labels JSON file')
    parser.add_argument('--model-path', type=str, default='models/trained/snowboard_lstm_improved.h5',
                       help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--target-timesteps', type=int, default=60,
                       help='Target timesteps for model input')
    parser.add_argument('--no-augmentation', action='store_true',
                       help='Disable data augmentation')
    
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
    trainer = ImprovedSnowboardTrainer(
        input_shape, 
        num_classes=len(np.unique(y_c)),
        use_augmentation=not args.no_augmentation
    )
    
    print(f"\n=== Training Configuration ===")
    print(f"Data shape: {X.shape}")
    print(f"Number of classes: {len(np.unique(y_c))}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Data augmentation: {not args.no_augmentation}")
    print(f"Model will be saved to: {args.model_path}")
    print()
    
    # 学習
    trainer.train(X, y_c, y_s, epochs=args.epochs, batch_size=args.batch_size)
    
    # 保存
    trainer.save_model(args.model_path)
    print(f"\nModel saved to {args.model_path}")


