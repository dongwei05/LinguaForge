# Direct curl download of unsloth/gemma-4-E4B-it-unsloth-bnb-4bit from hf-mirror.com
# Bypasses huggingface_hub's flaky Windows file locking.
#
# Usage: .\.venv\Scripts\python.exe vs powershell run this directly.
#   PS> powershell -File scripts\download_e4b_curl.ps1

$ErrorActionPreference = "Stop"

$REPO = "unsloth/gemma-4-E4B-it-unsloth-bnb-4bit"
$BASE = "https://hf-mirror.com/$REPO/resolve/main"
$DEST = "G:\models\gemma-4-E4B-bnb-4bit"

$files = @(
    "config.json",
    "generation_config.json",
    "processor_config.json",
    "chat_template.jinja",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "model-00001-of-00003.safetensors",
    "model-00002-of-00003.safetensors",
    "model-00003-of-00003.safetensors"
)

New-Item -ItemType Directory -Path $DEST -Force | Out-Null

foreach ($f in $files) {
    $out = Join-Path $DEST $f
    if ((Test-Path $out) -and ((Get-Item $out).Length -gt 1024)) {
        Write-Host "[SKIP] $f already exists ($([math]::Round((Get-Item $out).Length/1MB,1)) MB)"
        continue
    }
    Write-Host "[GET ] $f"
    # -L follow redirects, -C resume if partial, --retry on transient failure
    curl.exe -L -C - --retry 5 --retry-delay 3 -o $out "$BASE/$f"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "[FAIL] $f exit=$LASTEXITCODE" -ForegroundColor Red
        exit 1
    }
}

Write-Host "`n[OK] All files downloaded to $DEST" -ForegroundColor Green
Get-ChildItem $DEST -File | Format-Table @{n='Size_MB';e={[math]::Round($_.Length/1MB,1)}}, Name -AutoSize
