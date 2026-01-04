# 動画の一部を切り出すスクリプト（PowerShell版）
# 使い方: .\scripts\extract_video_segment.ps1 -InputVideo "input.mp4" -StartTime "00:00:30" -Duration 10 -OutputName "trick_001.mp4"

param(
    [Parameter(Mandatory=$true)]
    [string]$InputVideo,
    
    [Parameter(Mandatory=$true)]
    [string]$StartTime,  # 例: "00:00:30"
    
    [Parameter(Mandatory=$true)]
    [int]$Duration,  # 秒数
    
    [Parameter(Mandatory=$true)]
    [string]$OutputName
)

$OutputDir = "data\input"

# ディレクトリ作成
if (-not (Test-Path $OutputDir)) {
    New-Item -ItemType Directory -Path $OutputDir | Out-Null
}

$OutputPath = Join-Path $OutputDir $OutputName

# ffmpegがインストールされているか確認
$ffmpeg = Get-Command ffmpeg -ErrorAction SilentlyContinue

if (-not $ffmpeg) {
    Write-Host "ffmpeg is not installed. Using Python/OpenCV instead..." -ForegroundColor Yellow
    Write-Host ""
    
    # Pythonスクリプトを使用
    python scripts/extract_video_segment_python.py $InputVideo $StartTime $Duration $OutputName --output-dir $OutputDir
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Video extracted successfully: $OutputPath" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to extract video" -ForegroundColor Red
        exit 1
    }
} else {
    # ffmpegコマンド実行
    Write-Host "Extracting video segment..." -ForegroundColor Cyan
    Write-Host "  Input: $InputVideo" -ForegroundColor Gray
    Write-Host "  Start: $StartTime" -ForegroundColor Gray
    Write-Host "  Duration: $Duration seconds" -ForegroundColor Gray
    Write-Host "  Output: $OutputPath" -ForegroundColor Gray
    Write-Host ""

    ffmpeg -i $InputVideo -ss $StartTime -t $Duration -c copy $OutputPath

    if ($LASTEXITCODE -eq 0) {
        Write-Host "✓ Video extracted successfully: $OutputPath" -ForegroundColor Green
    } else {
        Write-Host "✗ Failed to extract video" -ForegroundColor Red
        exit 1
    }
}

