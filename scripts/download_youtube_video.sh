#!/bin/bash
# YouTube動画をダウンロードするスクリプト（Bash版）
# 使い方: bash scripts/download_youtube_video.sh "https://www.youtube.com/watch?v=VIDEO_ID" tutorial.mp4

if [ $# -lt 1 ]; then
    echo "Usage: $0 <youtube_url> [output_name]"
    echo "Example: $0 'https://www.youtube.com/watch?v=VIDEO_ID' tutorial.mp4"
    exit 1
fi

URL=$1
OUTPUT_NAME=${2:-"downloaded.mp4"}

# yt-dlpがインストールされているか確認
if ! command -v yt-dlp &> /dev/null; then
    echo "Error: yt-dlp is not installed."
    echo "Please install yt-dlp first:"
    echo "  pip install yt-dlp"
    exit 1
fi

echo "Downloading video from YouTube..."
echo "  URL: $URL"
echo "  Output: $OUTPUT_NAME"
echo ""

# 最高画質のMP4形式でダウンロード
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" "$URL" -o "$OUTPUT_NAME"

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Video downloaded successfully: $OUTPUT_NAME"
    echo ""
    echo "Next steps:"
    echo "  1. Extract segments using ffmpeg or extract_video_segment.sh"
    echo "  2. Place extracted videos in data/input/"
    echo "  3. Run batch_extract_features.sh to extract features"
else
    echo ""
    echo "✗ Failed to download video"
    exit 1
fi



