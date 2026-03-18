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
import subprocess
import sys
import time


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
                    print(f"Attempt {attempt + 1} of {max_retries} failed: {e}")
                    if attempt < max_retries - 1 and delay is not None:
                        if exponential_backoff:
                            backoff_delay = delay * (backoff_multiplier ** attempt)
                            if max_delay is not None:
                                backoff_delay = min(backoff_delay, max_delay)
                            print(f"Waiting {backoff_delay:.2f} seconds before retry")
                            time.sleep(backoff_delay)
                        else:
                            print(f"Waiting {delay} seconds before retry")
                            time.sleep(delay)
                    else:
                        raise e
        return wrapper
    return retry_decorator


def shell(cmd, env=None, cwd=None, out_format="plain"):
    """
    Run command execution in specified environment
    :param cmd: list containing command and its parameters
    :param env: set of environment variables to set for this command
    :param cwd: working directory from which execute call
    :param out_format: 'plain' or 'html'. If 'html' all '\n; symbols are replaced by '<br>' tag
    :return:
    """
    if sys.platform.startswith('linux') or sys.platform == 'darwin':
        cmd = ['/bin/bash', '-c', " ".join(cmd)]
    else:
        cmd = " ".join(cmd)

    sys.stdout.write("Running command:\n" + " ".join(cmd) + "\n")
    p = subprocess.Popen(cmd, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    (stdout, stderr) = p.communicate()
    stdout = str(stdout.decode('utf-8'))
    stderr = str(stderr.decode('utf-8'))
    if out_format == "html":
        stdout = "<br>\n".join(stdout.split('\n'))
        stderr = "<br>\n".join(stderr.split('\n'))
    return p.returncode, stdout, stderr
