"""
データ拡張モジュール
既存の特徴量データを変形してデータを増やす（教師データを増やさずに精度向上）
"""
import numpy as np


class DataAugmentation:
    """特徴量データの拡張クラス"""
    
    def __init__(self):
        pass
    
    def add_noise(self, features, noise_level=0.01):
        """
        ガウシアンノイズを追加
        
        Args:
            features: (time_steps, num_features) の特徴量
            noise_level: ノイズの強度（標準偏差）
        
        Returns:
            ノイズを追加した特徴量
        """
        noise = np.random.normal(0, noise_level, features.shape)
        return features + noise
    
    def time_shift(self, features, shift_range=5):
        """
        時間軸方向にシフト（前後数フレームずらす）
        
        Args:
            features: (time_steps, num_features) の特徴量
            shift_range: シフトするフレーム数の範囲
        
        Returns:
            シフトした特徴量
        """
        shift = np.random.randint(-shift_range, shift_range + 1)
        if shift == 0:
            return features
        
        shifted = np.roll(features, shift, axis=0)
        
        # シフトした部分を0埋めまたは端の値で埋める
        if shift > 0:
            shifted[:shift] = features[0]
        else:
            shifted[shift:] = features[-1]
        
        return shifted
    
    def scale_features(self, features, scale_range=(0.9, 1.1)):
        """
        特徴量をスケーリング（拡大・縮小）
        
        Args:
            features: (time_steps, num_features) の特徴量
            scale_range: スケールの範囲 (min, max)
        
        Returns:
            スケールした特徴量
        """
        scale = np.random.uniform(scale_range[0], scale_range[1])
        return features * scale
    
    def time_warp(self, features, warp_factor=0.1):
        """
        時間軸方向のワープ（時間の伸縮）
        
        Args:
            features: (time_steps, num_features) の特徴量
            warp_factor: ワープの強度
        
        Returns:
            ワープした特徴量
        """
        time_steps = features.shape[0]
        
        # ランダムなワープポイントを生成
        warp_points = np.random.uniform(-warp_factor, warp_factor, time_steps)
        warp_points = np.cumsum(1 + warp_points)
        warp_points = (warp_points / warp_points[-1]) * (time_steps - 1)
        
        # 線形補間（scipyなしで実装）
        original_indices = np.arange(time_steps)
        warped_features = np.zeros_like(features)
        
        for i in range(features.shape[1]):
            # 線形補間を手動で実装
            for j, wp in enumerate(warp_points):
                idx = int(wp)
                frac = wp - idx
                
                if idx >= time_steps - 1:
                    warped_features[j, i] = features[-1, i]
                elif idx < 0:
                    warped_features[j, i] = features[0, i]
                else:
                    warped_features[j, i] = (features[idx, i] * (1 - frac) + 
                                            features[idx + 1, i] * frac)
        
        return warped_features
    
    def augment(self, features, methods=['noise', 'time_shift', 'scale']):
        """
        複数の拡張手法を適用
        
        Args:
            features: (time_steps, num_features) の特徴量
            methods: 適用する拡張手法のリスト
        
        Returns:
            拡張された特徴量
        """
        augmented = features.copy()
        
        if 'noise' in methods:
            augmented = self.add_noise(augmented)
        
        if 'time_shift' in methods:
            augmented = self.time_shift(augmented)
        
        if 'scale' in methods:
            augmented = self.scale_features(augmented)
        
        # time_warpは計算コストが高いので、必要に応じて個別に使用
        
        return augmented
    
    def augment_batch(self, X, y_class, y_success, augmentation_factor=2):
        """
        バッチ全体を拡張
        
        Args:
            X: (samples, time_steps, features) の特徴量データ
            y_class: (samples,) の技分類ラベル
            y_success: (samples,) の成功判定ラベル
            augmentation_factor: 各サンプルを何倍に拡張するか
        
        Returns:
            拡張されたデータセット
        """
        X_augmented = []
        y_class_augmented = []
        y_success_augmented = []
        
        for i in range(len(X)):
            # 元のデータを追加
            X_augmented.append(X[i])
            y_class_augmented.append(y_class[i])
            y_success_augmented.append(y_success[i])
            
            # 拡張データを追加
            for _ in range(augmentation_factor - 1):
                # ランダムに拡張手法を選択
                methods = np.random.choice(
                    ['noise', 'time_shift', 'scale'],
                    size=np.random.randint(1, 4),
                    replace=False
                )
                
                aug_features = self.augment(X[i], methods=list(methods))
                X_augmented.append(aug_features)
                y_class_augmented.append(y_class[i])
                y_success_augmented.append(y_success[i])
        
        return (
            np.array(X_augmented),
            np.array(y_class_augmented),
            np.array(y_success_augmented)
        )


if __name__ == "__main__":
    # テスト
    aug = DataAugmentation()
    
    # サンプルデータ
    features = np.random.rand(60, 12)
    
    # 拡張テスト
    print("Original shape:", features.shape)
    
    noisy = aug.add_noise(features)
    print("Noise added shape:", noisy.shape)
    
    shifted = aug.time_shift(features)
    print("Time shifted shape:", shifted.shape)
    
    scaled = aug.scale_features(features)
    print("Scaled shape:", scaled.shape)

