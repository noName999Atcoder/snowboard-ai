"""
動画の一部を切り出すスクリプト（Python版 - ffmpeg不要）
OpenCVを使用して動画を切り出します
"""
import cv2
import os
import sys
import argparse
from pathlib import Path


def extract_video_segment(input_video, start_time, duration, output_path):
    """
    動画の一部を切り出す
    
    Args:
        input_video: 入力動画のパス
        start_time: 開始時刻（秒）
        duration: 切り出す長さ（秒）
        output_path: 出力ファイルのパス
    """
    if not os.path.exists(input_video):
        print(f"Error: Input video not found: {input_video}")
        return False
    
    # 出力ディレクトリを作成
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 動画を開く
    cap = cv2.VideoCapture(input_video)
    if not cap.isOpened():
        print(f"Error: Could not open video: {input_video}")
        return False
    
    # 動画の情報を取得
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 開始フレームと終了フレームを計算
    start_frame = int(start_time * fps)
    end_frame = int((start_time + duration) * fps)
    
    if start_frame >= total_frames:
        print(f"Error: Start time ({start_time}s) is beyond video length")
        cap.release()
        return False
    
    if end_frame > total_frames:
        end_frame = total_frames
        duration = (end_frame - start_frame) / fps
        print(f"Warning: Duration adjusted to {duration:.2f} seconds")
    
    # 開始フレームに移動
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
    # 出力動画の設定
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        print(f"Error: Could not create output video: {output_path}")
        cap.release()
        return False
    
    print(f"Extracting video segment...")
    print(f"  Input: {input_video}")
    print(f"  Start: {start_time}s (frame {start_frame})")
    print(f"  Duration: {duration}s (frames {end_frame - start_frame})")
    print(f"  Output: {output_path}")
    print()
    
    frame_count = start_frame
    while frame_count < end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        
        out.write(frame)
        frame_count += 1
        
        if frame_count % (fps * 5) == 0:  # 5秒ごとに進捗表示
            elapsed = (frame_count - start_frame) / fps
            print(f"  Progress: {elapsed:.1f}s / {duration:.1f}s", end='\r')
    
    cap.release()
    out.release()
    
    print(f"\n✓ Video extracted successfully: {output_path}")
    return True


def parse_time(time_str):
    """
    時刻文字列（HH:MM:SS または秒数）を秒数に変換
    
    Args:
        time_str: 時刻文字列（例: "00:18:00" または "1080"）
    
    Returns:
        秒数（float）
    """
    if ':' in time_str:
        # HH:MM:SS 形式
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(float, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(float, parts)
            return minutes * 60 + seconds
    else:
        # 秒数形式
        return float(time_str)


def main():
    parser = argparse.ArgumentParser(description='Extract video segment using OpenCV (ffmpeg not required)')
    parser.add_argument('input_video', type=str, help='Input video file path')
    parser.add_argument('start_time', type=str, help='Start time (HH:MM:SS or seconds)')
    parser.add_argument('duration', type=float, help='Duration in seconds')
    parser.add_argument('output_name', type=str, help='Output file name')
    parser.add_argument('--output-dir', type=str, default='data/input', help='Output directory')
    
    args = parser.parse_args()
    
    # 時刻を秒数に変換
    start_seconds = parse_time(args.start_time)
    
    # 出力パス
    output_path = os.path.join(args.output_dir, args.output_name)
    
    # 切り出し実行
    success = extract_video_segment(
        args.input_video,
        start_seconds,
        args.duration,
        output_path
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()



