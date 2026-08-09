<#
  一键金句切片：转录 → 找金句 → 生成挑选页。
  用法（在 D:\stream-tools 目录下打开 PowerShell）：
    .\make_highlights.ps1 -Video "C:\path\to\video.mp4"
  可选参数： -MaxClips 8  -MinSeconds 45  -MaxSeconds 75  -Language zh
  跑完看报告，选好编号后按提示运行剪辑命令。
#>
param(
  [Parameter(Mandatory = $true)][string]$Video,
  [int]$MaxClips = 8,
  [int]$MinSeconds = 45,
  [int]$MaxSeconds = 75,
  [int]$ChunkSeconds = 480,
  [string]$Language = "zh"
)
$ErrorActionPreference = "Stop"

if (-not (Test-Path -LiteralPath $Video)) { throw "找不到视频：$Video" }
$root  = $PSScriptRoot
$skill = Join-Path $root ".agents\skills\make-highlight-clips\scripts"
$dir   = Split-Path -Parent $Video
$stem  = [IO.Path]::GetFileNameWithoutExtension($Video)
$t     = Join-Path $dir "${stem}_highlights_transcript.json"
$h     = Join-Path $dir "${stem}_highlights.json"
$md    = Join-Path $dir "${stem}_highlights.md"
$html  = Join-Path $dir "${stem}_highlights_review.html"

Write-Host "① 转录中（3 小时视频约 20+ 个切片，请耐心）..." -ForegroundColor Cyan
python (Join-Path $skill "transcribe_fine.py") "$Video" -o "$t" --language $Language --chunk-seconds $ChunkSeconds

Write-Host "② 找金句中..." -ForegroundColor Cyan
python (Join-Path $skill "find_highlights.py") "$t" -o "$h" `
  --max-clips $MaxClips --min-seconds $MinSeconds --max-seconds $MaxSeconds

Write-Host "③ 生成交互挑选页..." -ForegroundColor Cyan
python (Join-Path $skill "build_review.py") "$h" -o "$html"

Write-Host ""
Write-Host "完成 ✅" -ForegroundColor Green
Write-Host "  报告（金句/时间戳/完整对话）： $md"
Write-Host "  挑选页（浏览器打开勾选）：     $html"
Write-Host ""
Write-Host "选好编号后剪辑（把 1,3,5 换成你要的编号）：" -ForegroundColor Yellow
Write-Host "  python `"$(Join-Path $skill 'cut_clips.py')`" `"$h`" --pick 1,3,5 --media `"$Video`" --burn-subs"
Write-Host "  横屏原片 + 竖屏 9:16 会输出到： $(Join-Path $dir 'clips')"
