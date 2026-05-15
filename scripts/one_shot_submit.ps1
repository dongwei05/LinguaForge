# One-shot: push everything once HF Write token + GH auth are ready.
# Usage:
#   $env:HF_TOKEN_WRITE = "hf_xxx"     # write-scoped token
#   $env:GH_REPO_NAME   = "LinguaForge"  # name on github.com
#   ./scripts/one_shot_submit.ps1

param(
    [string]$HfToken = $env:HF_TOKEN_WRITE,
    [string]$GhRepoName = $env:GH_REPO_NAME
)

if (-not $HfToken) { throw "HF_TOKEN_WRITE not set" }
if (-not $GhRepoName) { $GhRepoName = "LinguaForge" }
if (-not $env:GH_USER) { $env:GH_USER = (gh api user --jq .login 2>$null) }

$env:HF_TOKEN = $HfToken
$env:PYTHONUTF8 = "1"
$env:PYTHONIOENCODING = "utf-8"
$env:Path += ";C:\Program Files\GitHub CLI"

Write-Host "=== Step 1/3: push LoRA adapter to HF Hub ==="
python scripts/push_to_hf.py
if ($LASTEXITCODE -ne 0) { throw "push_to_hf.py failed" }

Write-Host ""
Write-Host "=== Step 2/3: create + push GitHub repo ==="
$ghUser = (gh api user --jq .login 2>$null)
if (-not $ghUser) { throw "gh CLI not authenticated; run 'gh auth login --web' first" }
$existing = (gh repo view "$ghUser/$GhRepoName" 2>$null)
if ($existing) {
    Write-Host "Repo $ghUser/$GhRepoName already exists; setting it as remote and pushing."
    git remote remove origin 2>$null | Out-Null
    git remote add origin "https://github.com/$ghUser/$GhRepoName.git"
    git push -u origin main
} else {
    gh repo create "$ghUser/$GhRepoName" --public --source . --remote origin --push --description "LinguaForge - Gemma 4 LoRA across 204 endangered/low-resource languages"
}
if ($LASTEXITCODE -ne 0) { throw "GitHub push failed" }

Write-Host ""
Write-Host "=== Step 3/3: deploy Hugging Face Space ==="
python scripts/push_to_hf_space.py
if ($LASTEXITCODE -ne 0) { throw "Space deploy failed" }

Write-Host ""
Write-Host "=== ALL DONE ==="
Write-Host "  Repo:    https://github.com/$ghUser/$GhRepoName"
Write-Host "  Adapter: https://huggingface.co/zcgf111/linguaforge-gemma4-204lang-lora"
Write-Host "  Space:   https://huggingface.co/spaces/zcgf111/LinguaForge"
Write-Host ""
Write-Host "Next: walk through SUBMISSION_CHECKLIST.md."
