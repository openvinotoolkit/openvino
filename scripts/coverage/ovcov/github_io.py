# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path
import os


class GithubIO:
    def __init__(self, workspace: Path) -> None:
        self.workspace = workspace
        self._github_env = os.environ.get("GITHUB_ENV", "").strip()
        self._github_summary = os.environ.get("GITHUB_STEP_SUMMARY", "").strip()

        self.local_state_dir = workspace / ".tmp" / "coverage-local"
        self.local_env_file = self.local_state_dir / "github_env"
        self.local_summary_file = self.local_state_dir / "step_summary.md"

        self.local_state_dir.mkdir(parents=True, exist_ok=True)
        self.local_env_file.touch(exist_ok=True)
        self.local_summary_file.touch(exist_ok=True)

    @property
    def env_file(self) -> Path:
        if self._github_env:
            return Path(self._github_env)
        return self.local_env_file

    @property
    def summary_file(self) -> Path:
        if self._github_summary:
            return Path(self._github_summary)
        return self.local_summary_file

    @property
    def is_github_mode(self) -> bool:
        return bool(self._github_env)

    def load_local_env(self) -> dict[str, str]:
        if self.is_github_mode:
            return {}

        loaded: dict[str, str] = {}
        if not self.local_env_file.exists():
            return loaded

        for raw_line in self.local_env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, value = line.split("=", 1)
            loaded[key] = value
        return loaded

    def export_env(self, key: str, value: str) -> None:
        with self.env_file.open("a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
        os.environ[key] = value

    def append_summary(self, text: str) -> None:
        with self.summary_file.open("a", encoding="utf-8") as f:
            f.write(text)
            if not text.endswith("\n"):
                f.write("\n")
