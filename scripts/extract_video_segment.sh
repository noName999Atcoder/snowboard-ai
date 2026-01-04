#!/bin/bash
# 動画の一部を切り出すスクリプト（Bash版）
# 使い方: bash scripts/extract_video_segment.sh input.mp4 00:00:30 10 trick_001.mp4

if [ $# -lt 4 ]; then
    echo "Usage: $0 <input_video> <start_time> <duration_seconds> <output_name>"
    echo "Example: $0 input.mp4 00:00:30 10 trick_001.mp4"
    exit 1
fi

INPUT_VIDEO=$1
START_TIME=$2
DURATION=$3
OUTPUT_NAME=$4

OUTPUT_DIR="data/input"

# ディレクトリ作成
mkdir -p "$OUTPUT_DIR"

OUTPUT_PATH="$OUTPUT_DIR/$OUTPUT_NAME"

# ffmpegコマンド実行
echo "Extracting video segment..."
echo "  Input: $INPUT_VIDEO"
echo "  Start: $START_TIME"
echo "  Duration: $DURATION seconds"
echo "  Output: $OUTPUT_PATH"
echo ""

ffmpeg -i "$INPUT_VIDEO" -ss "$START_TIME" -t "$DURATION" -c copy "$OUTPUT_PATH"

if [ $? -eq 0 ]; then
    echo "✓ Video extracted successfully: $OUTPUT_PATH"
else
    echo "✗ Failed to extract video"
    exit 1
fi



