#!/usr/bin/env python3
# Copyright (C) 2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

""" Common utilities for working with processes.
"""

import errno
import os
import logging
import subprocess
import sys


def get_env_from(script):
    """ Get environment set by a shell script
    """
    if not os.path.exists(str(script)):
        raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), str(script))
    env = {}
    if sys.platform == "win32":
        cmd = f'"{script}" && set'
    else:
        cmd = f'source "{script}" && env'
    dump = subprocess.check_output(cmd, shell=True, universal_newlines=True).strip()
    for line in dump.split("\n"):
        # split by first '='
        pair = [str(val).strip() for val in line.split("=", 1)]
        if len(pair) > 1 and pair[0]:  # ignore invalid entries
            env[pair[0]] = pair[1]
    return env


def cmd_exec(args, env=None, log=None, verbose=True):
    """ Run cmd using subprocess with logging and other improvements
    """
    if log is None:
        log = logging.getLogger()
    log_out = log.info if verbose else log.debug

    log_out(  # pylint: disable=logging-fstring-interpolation
        f'========== cmd: {" ".join(args)}'
    )

    proc = subprocess.Popen(
        args,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding="utf-8",
        universal_newlines=True,
    )
    output = []
    for line in iter(proc.stdout.readline, ""):
        log_out(line.strip("\n"))
        output.append(line)
        if line or proc.poll() is None:
            continue
        break
    outs = proc.communicate()[0]

    if outs:
        log_out(outs.strip("\n"))
        output.append(outs)
    log_out("========== Completed. Exit code: %d", proc.returncode)
    return proc.returncode, "".join(output)
