# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os


def pytest_configure(config):
    """Redirect each pytest-xdist worker to its own log file.

    When running with --numprocesses, each worker process inherits the same
    --log-file path from the CLI. Without this hook all workers write to a
    single file (interleaved / race-prone) and, critically, when a worker is
    killed (e.g. by pytest-timeout) the still-buffered records are lost.

    This hook renames the log file to include the worker ID so every worker
    writes independently: samples_smoke_tests_gw0.log, _gw1.log, etc.
    The main process keeps the original path for the overall pytest summary.
    """
    worker_id = os.environ.get("PYTEST_XDIST_WORKER")
    if worker_id is None:
        return  # main process – --log-file CLI arg already applies
    log_file = getattr(config.option, "log_file", None)
    if not log_file:
        return
    base, ext = os.path.splitext(log_file)
    config.option.log_file = f"{base}_{worker_id}{ext}"
