"""
改善された特徴量抽出
より多くの特徴量を抽出して精度向上
"""
import numpy as np
import math
from extract_features import FeatureExtractor


class ImprovedFeatureExtractor(FeatureExtractor):
    """改善された特徴量抽出クラス"""
    
    def extract(self, keypoints_sequence):
        """
        時系列キーポイントデータから特徴量を抽出する（改善版）
        Args:
            keypoints_sequence: (frames, 1, 17, 3)
        Returns:
            features: (frames, num_features)
        """
        features_list = []
        
        for i, frame_data in enumerate(keypoints_sequence):
            # (1, 17, 3) -> (17, 3)
            kps = np.squeeze(frame_data)
            
            # 座標取得 (y, x, score)
            ls = kps[self.KEYPOINT_DICT['left_shoulder']][:2]
            rs = kps[self.KEYPOINT_DICT['right_shoulder']][:2]
            lh = kps[self.KEYPOINT_DICT['left_hip']][:2]
            rh = kps[self.KEYPOINT_DICT['right_hip']][:2]
            lk = kps[self.KEYPOINT_DICT['left_knee']][:2]
            la = kps[self.KEYPOINT_DICT['left_ankle']][:2]
            rk = kps[self.KEYPOINT_DICT['right_knee']][:2]
            ra = kps[self.KEYPOINT_DICT['right_ankle']][:2]
            nose = kps[self.KEYPOINT_DICT['nose']][:2]
            
            # 基本特徴量（元の実装）
            shoulder_slope = self.calculate_slope(ls, rs)
            hip_slope = self.calculate_slope(lh, rh)
            twist = shoulder_slope - hip_slope
            left_knee_angle = self.calculate_angle(lh, lk, la)
            right_knee_angle = self.calculate_angle(rh, rk, ra)
            center_hip_y = (lh[0] + rh[0]) / 2
            
            # 追加特徴量1: 肩と腰の距離（体の開き具合）
            shoulder_center = (ls + rs) / 2
            hip_center = (lh + rh) / 2
            shoulder_hip_distance = np.linalg.norm(shoulder_center - hip_center)
            
            # 追加特徴量2: 体の傾き（肩と腰の中点を結ぶ線の角度）
            body_tilt = self.calculate_slope(shoulder_center, hip_center)
            
            # 追加特徴量3: 左右の膝の角度差（バランス）
            knee_angle_diff = abs(left_knee_angle - right_knee_angle)
            
            # 追加特徴量4: 重心位置（肩、腰、膝の重み付き平均）
            center_of_mass_y = (shoulder_center[0] * 0.2 + hip_center[0] * 0.5 + 
                              (lk[0] + rk[0]) / 2 * 0.3)
            
            # 追加特徴量5: 体の幅（肩幅と腰幅）
            shoulder_width = np.linalg.norm(ls - rs)
            hip_width = np.linalg.norm(lh - rh)
            
            # 追加特徴量6: 頭の位置（nose）の高さ
            head_height = nose[0]
            
            # 追加特徴量7: 左右のバランス（肩と腰の左右差）
            left_side_angle = self.calculate_angle(ls, lh, lk)
            right_side_angle = self.calculate_angle(rs, rh, rk)
            side_balance = abs(left_side_angle - right_side_angle)
            
            features_list.append([
                shoulder_slope,
                hip_slope,
                twist,
                left_knee_angle,
                right_knee_angle,
                center_hip_y,
                # 追加特徴量
                shoulder_hip_distance,
                body_tilt,
                knee_angle_diff,
                center_of_mass_y,
                shoulder_width,
                hip_width,
                head_height,
                side_balance
            ])
        
        features = np.array(features_list)
        
        # 時間変化量を追加
        diffs = np.diff(features, axis=0, prepend=features[0:1])
        
        # 加速度（2次微分）も追加
        if len(features) > 2:
            accels = np.diff(diffs, axis=0, prepend=diffs[0:1])
        else:
            accels = np.zeros_like(diffs)
        
        # 特徴量結合
        combined_features = np.hstack([features, diffs, accels])
        return combined_features

