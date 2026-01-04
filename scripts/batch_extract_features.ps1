# バッチ特徴量抽出スクリプト（PowerShell版）
# data/input/ 内のすべての動画ファイルから特徴量を抽出します

$inputDir = "data\input"
$outputDir = "data\features"

# ディレクトリの存在確認
if (-not (Test-Path $inputDir)) {
    Write-Host "Error: Input directory not found: $inputDir" -ForegroundColor Red
    exit 1
}

# 動画ファイルを取得
$videoFiles = Get-ChildItem -Path $inputDir -Filter *.mp4

if ($videoFiles.Count -eq 0) {
    Write-Host "No video files found in $inputDir" -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($videoFiles.Count) video file(s)" -ForegroundColor Green
Write-Host ""

# 各動画ファイルを処理
foreach ($video in $videoFiles) {
    Write-Host "Processing: $($video.Name)" -ForegroundColor Cyan
    python src/preprocessing/extract_and_save_features.py --input $video.FullName --output-dir $outputDir
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "  ✓ Completed: $($video.Name)" -ForegroundColor Green
    } else {
        Write-Host "  ✗ Failed: $($video.Name)" -ForegroundColor Red
    }
    Write-Host ""
}

Write-Host "Batch processing completed!" -ForegroundColor Green




