# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Usage: append_variables.ps1 -TokenPath <path>
# Reads a HuggingFace token from a file, masks it in GitHub Actions logs,
# and appends HF_TOKEN to GITHUB_ENV for use in subsequent steps.

param(
    [Parameter(Mandatory = $true)]
    [string]$TokenPath
)

if (-not (Test-Path $TokenPath)) {
    Write-Warning "⚠ HuggingFace token file not found at $TokenPath, skipping"
    exit 0
}

$HF_TOKEN = (Get-Content $TokenPath -Raw).Trim()

Write-Output "::add-mask::$HF_TOKEN"
Add-Content -Path $env:GITHUB_ENV -Value "HF_TOKEN=$HF_TOKEN"
Write-Output "✓ HuggingFace token loaded and masked"
