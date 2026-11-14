# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
 Copyright (C) 2018-2026 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""
import functools
import logging
import os
import subprocess
import sys
import time

_log = logging.getLogger(__name__)


def retry(max_retries=3, exceptions=(Exception,), delay=None, exponential_backoff=False, backoff_multiplier=2, max_delay=None):
    """
    Retry decorator with optional exponential backoff.

    Args:
        max_retries: Maximum number of retry attempts
        exceptions: Tuple of exception types to catch and retry on
        delay: Base delay in seconds between retries
        exponential_backoff: If True, use exponential backoff instead of fixed delay
        backoff_multiplier: Multiplier for exponential backoff (default: 2)
        max_delay: Maximum delay cap for exponential backoff
    """
    def retry_decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    _log.warning("Attempt %d of %d failed: %s", attempt + 1, max_retries, e)
                    if attempt < max_retries - 1 and delay is not None:
                        if exponential_backoff:
                            backoff_delay = delay * (backoff_multiplier ** attempt)
                            if max_delay is not None:
                                backoff_delay = min(backoff_delay, max_delay)
                            _log.debug("Waiting %.2f seconds before retry", backoff_delay)
                            time.sleep(backoff_delay)
                        else:
                            _log.debug("Waiting %s seconds before retry", delay)
                            time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return retry_decorator


def shell(cmd, env=None, cwd=None, out_format="plain", timeout=290):
    """
    Run command execution in specified environment
    :param cmd: list containing command and its parameters
    :param env: set of environment variables to set for this command
    :param cwd: working directory from which execute call
    :param out_format: 'plain' or 'html'. If 'html' all '\n; symbols are replaced by '<br>' tag
    :param timeout: seconds to wait before killing the process, or None to wait indefinitely.
                    Default is 290 s — just under the pytest-timeout of 300 s so that a hanging
                    sample binary is killed and partial output is captured *before* pytest-timeout
                    fires.  On Windows the thread-based pytest-timeout method cannot interrupt
                    a blocking communicate() call, so the timeout must be enforced here.
    :return:
    """

    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', " ".join(cmd)]
    else:
        cmd = " ".join(cmd)

    cmd_str = " ".join(cmd) if isinstance(cmd, list) else cmd
    _log.debug("Running: %s", cmd_str)
    # Always set PYTHONUNBUFFERED so Python subprocesses flush stdout to the pipe
    # immediately — without this, stdout is fully buffered when writing to a PIPE
    # and we see no output at all from a hanging Python sample process.
    effective_env = dict(os.environ if env is None else env)
    effective_env.setdefault("PYTHONUNBUFFERED", "1")
    p = subprocess.Popen(cmd, cwd=cwd, env=effective_env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    _log.debug("PID %d started", p.pid)
    try:
        try:
            stdout, stderr = p.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            p.kill()
            stdout, stderr = p.communicate()
            stdout_str = stdout.decode("utf-8") if stdout else ""
            _log.warning("PID %d timed out after %s s; killed. stdout: %s", p.pid, timeout, stdout_str[:2000])
            return -1, stdout_str, f"Command timed out after {timeout} seconds"
    finally:
        if p.poll() is None:
            _log.warning(
                "PID %d still running (likely pytest-timeout); killing.\n  cmd: %s",
                p.pid,
                cmd_str,
            )
            p.kill()
            try:
                partial_out, partial_err = p.communicate(timeout=10)
                _log.warning(
                    "PID %d partial output before kill:\nstdout: %s\nstderr: %s",
                    p.pid,
                    partial_out.decode("utf-8", errors="replace")[:4000] if partial_out else "<empty>",
                    partial_err.decode("utf-8", errors="replace")[:2000] if partial_err else "<empty>",
                )
            except Exception:
                p.wait()
    stdout = str(stdout.decode('utf-8'))
    stderr = str(stderr.decode('utf-8'))
    _log.debug("PID %d exited %d\nstdout: %s\nstderr: %s", p.pid, p.returncode, stdout[:4000], stderr[:4000])
    if out_format == "html":
        stdout = "<br>\n".join(stdout.split('\n'))
        stderr = "<br>\n".join(stderr.split('\n'))
    return p.returncode, stdout, stderr
