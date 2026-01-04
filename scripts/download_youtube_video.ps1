# YouTube動画をダウンロードするスクリプト（PowerShell版）
# 使い方: .\scripts\download_youtube_video.ps1 -Url "https://www.youtube.com/watch?v=VIDEO_ID" -OutputName "tutorial.mp4"

param(
    [Parameter(Mandatory=$true)]
    [string]$Url,
    
    [Parameter(Mandatory=$false)]
    [string]$OutputName = "downloaded.mp4"
)

# yt-dlpがインストールされているか確認
$ytdlp = Get-Command yt-dlp -ErrorAction SilentlyContinue

if (-not $ytdlp) {
    Write-Host "Error: yt-dlp is not installed." -ForegroundColor Red
    Write-Host "Please install yt-dlp first:" -ForegroundColor Yellow
    Write-Host "  pip install yt-dlp" -ForegroundColor Cyan
    exit 1
}

Write-Host "Downloading video from YouTube..." -ForegroundColor Cyan
Write-Host "  URL: $Url" -ForegroundColor Gray
Write-Host "  Output: $OutputName" -ForegroundColor Gray
Write-Host ""

# 最高画質のMP4形式でダウンロード
yt-dlp -f "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]" $Url -o $OutputName

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "✓ Video downloaded successfully: $OutputName" -ForegroundColor Green
    Write-Host ""
    Write-Host "Next steps:" -ForegroundColor Yellow
    Write-Host "  1. Extract segments using ffmpeg or extract_video_segment.ps1" -ForegroundColor Gray
    Write-Host "  2. Place extracted videos in data/input/" -ForegroundColor Gray
    Write-Host "  3. Run batch_extract_features.ps1 to extract features" -ForegroundColor Gray
} else {
    Write-Host ""
    Write-Host "✗ Failed to download video" -ForegroundColor Red
    exit 1
}


