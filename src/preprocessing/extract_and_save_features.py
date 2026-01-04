"""
動画から特徴量を抽出して保存するスクリプト
再学習フロー用：動画を処理して特徴量をnumpyファイルとして保存する
"""
import cv2
import numpy as np
import os
import argparse
import sys

# プロジェクトルートをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(current_dir)
sys.path.insert(0, src_dir)

from pose.movenet import MoveNet
from preprocessing.extract_features import FeatureExtractor


def extract_features_from_video(video_path, output_dir):
    """
    動画から特徴量を抽出して保存する
    
    Args:
        video_path: 入力動画のパス
        output_dir: 特徴量を保存するディレクトリ
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    # ディレクトリ作成
    os.makedirs(output_dir, exist_ok=True)
    
    # モジュール初期化
    print("Initializing MoveNet...")
    pose_estimator = MoveNet()
    feature_extractor = FeatureExtractor()
    
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
    
    extract_features_from_video(args.input, args.output_dir)


if __name__ == "__main__":
    main()

