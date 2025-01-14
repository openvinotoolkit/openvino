# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility module with config environment utilities."""

import os


def fix_path(path, root_path=None):
    """
    Fix path: expand environment variables if any, make absolute path from
    root_path/path if path is relative, resolve symbolic links encountered.
    """
    path = os.path.expandvars(path)
    if not os.path.isabs(path) and root_path is not None:
        path = os.path.join(root_path, path)
    return os.path.realpath(os.path.abspath(path))


def fix_env_conf(env, root_path=None):
    """Fix paths in environment config."""
    for name, value in env.items():
        if isinstance(value, dict):
            # if value is dict, think of it as of a (sub)environment
            # within current environment
            # since it can also contain envvars/relative paths,
            # recursively update (sub)environment as well
            env[name] = fix_env_conf(value, root_path=root_path)
        else:
            env[name] = fix_path(value, root_path=root_path)
    return env
