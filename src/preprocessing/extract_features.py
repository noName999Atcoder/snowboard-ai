
import cv2
import numpy as np
import os
import argparse
import math

from pose.movenet import MoveNet

class FeatureExtractor:
    """
    特徴量抽出クラス
    - 基本的な特徴量に加えて、追加の特徴量を抽出し、時間変化量（速度）と加速度も計算する。
    """
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
            
            # 基本特徴量
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
        
        # 時間変化量を追加 (速度)
        diffs = np.diff(features, axis=0, prepend=features[0:1])
        
        # 加速度（2次微分）も追加
        if len(features) > 2:
            accels = np.diff(diffs, axis=0, prepend=diffs[0:1])
        else:
            accels = np.zeros_like(diffs)
        
        # 特徴量結合
        combined_features = np.hstack([features, diffs, accels])
        return combined_features


def extract_features_from_video(video_path, output_dir, feature_extractor):
    """
    動画から特徴量を抽出して保存する
    
    Args:
        video_path: 入力動画のパス
        output_dir: 特徴量を保存するディレクトリ
        feature_extractor: 使用するFeatureExtractorのインスタンス
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # モジュール初期化
    print("Initializing MoveNet...")
    pose_estimator = MoveNet()
    
    # 動画読み込み
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return None
    
    print("Extracting keypoints from video...")
    all_keypoints = []
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 骨格抽出
        keypoints = pose_estimator.run_inference(frame)
        all_keypoints.append(keypoints)
        frame_count += 1
        
        if frame_count % 30 == 0:
            print(f"Processed {frame_count} frames...")
    
    cap.release()
    
    if len(all_keypoints) == 0:
        print("Error: No frames extracted from video")
        return None
    
    # numpy配列化
    all_keypoints_np = np.array(all_keypoints)
    
    # 特徴量抽出
    print("Extracting features...")
    features = feature_extractor.extract(all_keypoints_np)
    
    # ファイル名生成（動画ファイル名から）
    video_basename = os.path.splitext(os.path.basename(video_path))[0]
    output_path = os.path.join(output_dir, f"{video_basename}_features.npy")
    
    # 保存
    np.save(output_path, features)
    print(f"Features saved to: {output_path}")
    print(f"Feature shape: {features.shape}")
    
    return output_path


def main():
    parser = argparse.ArgumentParser(description='Extract features from video for training')
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output-dir', type=str, 
                       default='data/features',
                       help='Directory to save extracted features')
    args = parser.parse_args()
    
    # 絶対パスに変換
    if not os.path.isabs(args.input):
        args.input = os.path.abspath(args.input)
    if not os.path.isabs(args.output_dir):
        args.output_dir = os.path.abspath(args.output_dir)

    # 統合された最新のFeatureExtractorを使用
    feature_extractor = FeatureExtractor()
    extract_features_from_video(args.input, args.output_dir, feature_extractor)


if __name__ == "__main__":
    main()
