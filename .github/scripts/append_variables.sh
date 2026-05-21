#!/usr/bin/env bash
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# Usage: append_variables.sh <token-file-path>
# Reads a HuggingFace token from a file, masks it in GitHub Actions logs,
# and appends HF_TOKEN to GITHUB_ENV for use in subsequent steps.

set -euo pipefail

TOKEN_PATH="${1:?Usage: $0 <token-file-path>}"

if [[ ! -f "${TOKEN_PATH}" ]]; then
  echo "ERROR: Token file not found at: ${TOKEN_PATH}" >&2
  exit 1
fi

HF_TOKEN="$(cat "${TOKEN_PATH}")"
HF_TOKEN="${HF_TOKEN//[$'\r\n']}"  # strip any carriage return / newline

echo "::add-mask::${HF_TOKEN}"
echo "HF_TOKEN=${HF_TOKEN}" >> "${GITHUB_ENV}"
echo "✓ HuggingFace token loaded and masked"
