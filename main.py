import argparse
import cv2
import os
import numpy as np
import sys
import json

# srcディレクトリをパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.join(current_dir, 'src')
sys.path.insert(0, src_dir)

from pose.movenet import MoveNet
from preprocessing.extract_features import FeatureExtractor
from visualize.draw_pose import PoseVisualizer
from analysis.inference import TrickAnalyzer
from analysis.rules import RuleBasedFeedback

def load_trick_names(config_path="data/config/trick_classes.json"):
    """設定ファイルからトリック名とIDの辞書を読み込む"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # JSONの構造に合わせて 'display_name' を取得
        trick_names = {int(k): v['display_name'] for k, v in config['trick_classes'].items()}
        return trick_names
    except FileNotFoundError:
        print(f"Warning: Config file not found at {config_path}. Using default trick names.")
        return {0: "Nose Press", 1: "Tail Press", 2: "Butter 180"}
    except (json.JSONDecodeError, KeyError) as e:
        print(f"Warning: Error reading or parsing config file: {e}. Using default trick names.")
        return {0: "Nose Press", 1: "Tail Press", 2: "Butter 180"}

def main():
    parser = argparse.ArgumentParser(description='Snowboard AI Analysis Tool')
    parser.add_argument('--input', type=str, required=True, help='Path to input video')
    parser.add_argument('--output', type=str, default='data/output/output.mp4', help='Path to output video')
    parser.add_argument('--model', type=str, default='models/trained/snowboard_lstm.h5', help='Path to trained model')
    args = parser.parse_args()

    # 設定ファイルからトリック名を取得
    trick_names = load_trick_names()

    # パス設定
    input_path = args.input
    output_path = args.output
    
    if not os.path.exists(input_path):
        print(f"Error: Input file not found: {input_path}")
        return

    # モジュール初期化
    print("Initializing modules...")
    pose_estimator = MoveNet()
    feature_extractor = FeatureExtractor()
    visualizer = PoseVisualizer()
    analyzer = TrickAnalyzer(args.model)
    feedback_gen = RuleBasedFeedback()

    # 動画読み込み
    cap = cv2.VideoCapture(input_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # 出力設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print("Processing video...")
    
    all_keypoints = []
    frames_buffer = []

    # 1. 骨格抽出ループ
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # 推論
        keypoints = pose_estimator.run_inference(frame)
        all_keypoints.append(keypoints)
        frames_buffer.append(frame)

    cap.release()
    
    # numpy配列化 (frames, 1, 17, 3)
    all_keypoints_np = np.array(all_keypoints)
    
    # 2. 特徴量抽出
    print("Extracting features...")
    features = feature_extractor.extract(all_keypoints_np)
    
    # 3. AI解析 (時系列データ全体に対して1回の判定を行う想定)
    # ※本来はスライディングウィンドウなどでリアルタイム判定するが、
    # ここでは動画全体で1つの技と仮定して、リサイズまたはパディングして推論する簡易実装とする
    
    # モデル入力サイズに合わせてリサイズ (例: 60フレーム)
    target_timesteps = 60
    current_timesteps = features.shape[0]

    if current_timesteps == 0:
        print("Error: No features extracted. Cannot run inference.")
        return
    
    # 簡易的なリサンプリング
    indices = np.linspace(0, current_timesteps - 1, target_timesteps).astype(int)
    resampled_features = features[indices]
    
    print("Running inference...")
    trick_id, success_prob = analyzer.predict(resampled_features)
    
    trick_name = trick_names.get(trick_id, "Unknown") if trick_id is not None else "Model Not Loaded"
    
    # フィードバック生成
    advice = feedback_gen.generate_feedback(trick_id, success_prob if success_prob else 0, features)
    print(f"\n=== Analysis Result ===")
    print(f"Trick: {trick_name}")
    print(f"Success Probability: {success_prob:.2f}" if success_prob else "Success: N/A")
    print(f"Advice:\n{advice}")
    
    # 4. 結果描画と保存
    print("Saving output video...")
    for i, frame in enumerate(frames_buffer):
        # 骨格描画
        vis_frame = visualizer.draw(frame, all_keypoints_np[i])
        
        # テキスト描画
        cv2.putText(vis_frame, f"Trick: {trick_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(vis_frame)
        
    out.release()
    print("Done!")

if __name__ == "__main__":
    main()