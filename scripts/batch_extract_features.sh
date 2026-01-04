#!/bin/bash
# バッチ特徴量抽出スクリプト（Bash版）
# data/input/ 内のすべての動画ファイルから特徴量を抽出します

INPUT_DIR="data/input"
OUTPUT_DIR="data/features"

# ディレクトリの存在確認
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory not found: $INPUT_DIR"
    exit 1
fi

# 動画ファイルを取得
VIDEO_FILES=$(find "$INPUT_DIR" -name "*.mp4" -o -name "*.avi" -o -name "*.mov")

if [ -z "$VIDEO_FILES" ]; then
    echo "No video files found in $INPUT_DIR"
    exit 0
fi

# ファイル数をカウント
COUNT=$(echo "$VIDEO_FILES" | wc -l)
echo "Found $COUNT video file(s)"
echo ""

# 各動画ファイルを処理
for video in $VIDEO_FILES; do
    echo "Processing: $(basename "$video")"
    python src/preprocessing/extract_and_save_features.py --input "$video" --output-dir "$OUTPUT_DIR"
    
    if [ $? -eq 0 ]; then
        echo "  ✓ Completed: $(basename "$video")"
    else
        echo "  ✗ Failed: $(basename "$video")"
    fi
    echo ""
done

echo "Batch processing completed!"



