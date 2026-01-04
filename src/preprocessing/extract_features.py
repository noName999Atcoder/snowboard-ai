import numpy as np
import math

class FeatureExtractor:
    def __init__(self):
        # MoveNet Keypoint Indices
        self.KEYPOINT_DICT = {
            'nose': 0,
            'left_shoulder': 5, 'right_shoulder': 6,
            'left_elbow': 7, 'right_elbow': 8,
            'left_wrist': 9, 'right_wrist': 10,
            'left_hip': 11, 'right_hip': 12,
            'left_knee': 13, 'right_knee': 14,
            'left_ankle': 15, 'right_ankle': 16
        }

    def calculate_angle(self, a, b, c):
        """3点(a, b, c)から角度を計算する (bが中心)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
        angle = np.abs(radians*180.0/np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
        return angle

    def calculate_slope(self, a, b):
        """2点間の傾き（水平からの角度）を計算"""
        dy = b[0] - a[0] # y coordinate is at index 0 in MoveNet output
        dx = b[1] - a[1] # x coordinate is at index 1
        # 注意: MoveNetは (y, x) の順
        return np.degrees(np.arctan2(dy, dx))

    def extract(self, keypoints_sequence):
        """
        時系列キーポイントデータから特徴量を抽出する
        Args:
            keypoints_sequence: (frames, 1, 17, 3)
        Returns:
            features: (frames, num_features)
        """
        features_list = []
        
        for frame_data in keypoints_sequence:
            # (1, 17, 3) -> (17, 3)
            kps = np.squeeze(frame_data)
            
            # 座標取得 (y, x, score)
            # 信頼度が低い場合は直前の値を保持するなどの処理が必要だが、
            # ここでは簡易的にそのまま計算する
            
            ls = kps[self.KEYPOINT_DICT['left_shoulder']][:2]
            rs = kps[self.KEYPOINT_DICT['right_shoulder']][:2]
            lh = kps[self.KEYPOINT_DICT['left_hip']][:2]
            rh = kps[self.KEYPOINT_DICT['right_hip']][:2]
            lk = kps[self.KEYPOINT_DICT['left_knee']][:2]
            la = kps[self.KEYPOINT_DICT['left_ankle']][:2]
            rk = kps[self.KEYPOINT_DICT['right_knee']][:2]
            ra = kps[self.KEYPOINT_DICT['right_ankle']][:2]

            # 1. 肩の傾き
            shoulder_slope = self.calculate_slope(ls, rs)
            
            # 2. 腰の傾き
            hip_slope = self.calculate_slope(lh, rh)
            
            # 3. ねじれ (肩と腰の角度差)
            twist = shoulder_slope - hip_slope
            
            # 4. 膝の曲がり具合 (左)
            left_knee_angle = self.calculate_angle(lh, lk, la)
            
            # 5. 膝の曲がり具合 (右)
            right_knee_angle = self.calculate_angle(rh, rk, ra)
            
            # 重心位置（簡易：腰の中点）のY座標（高さ）
            center_hip_y = (lh[0] + rh[0]) / 2

            features_list.append([
                shoulder_slope,
                hip_slope,
                twist,
                left_knee_angle,
                right_knee_angle,
                center_hip_y
            ])
            
        features = np.array(features_list)
        
        # 6. 角速度などの時間変化量を追加 (diff)
        # 先頭は0埋め
        diffs = np.diff(features, axis=0, prepend=features[0:1])
        
        # 特徴量結合
        combined_features = np.hstack([features, diffs])
        return combined_features