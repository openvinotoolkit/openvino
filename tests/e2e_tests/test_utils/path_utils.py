# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Path utils used across E2E tests framework."""

import datetime
import getpass
import hashlib
import logging as log
import math
import os
import re
import socket
import sys
import time
from contextlib import contextmanager
from glob import iglob
from pathlib import Path, PurePath
from typing import Union

log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)


@contextmanager
def import_from(path):
    """
    Import decorator to resolve module import issues
    """
    path = os.path.abspath(os.path.realpath(path))
    sys.path.insert(0, path)
    yield
    sys.path.remove(path)


def resolve_file_path(file: str, as_str=False):
    """Return absolute file path checking if it exists in the process."""
    path = Path(file).resolve()
    if not path.is_file():
        raise FileNotFoundError("{} doesn't exist".format(path))
    if as_str:
        return str(path)
    return path


def resolve_dir_path(file: str, as_str=False):
    """Return absolute directory path if it exists in the process."""
    path = Path(file).resolve()
    if not path.is_dir():
        raise FileNotFoundError("{} doesn't exist".format(path))
    if as_str:
        return str(path)
    return path


def is_absolute(path: str):
    """Check if given path is an absolute path."""
    return Path(path).is_absolute()


def is_writable(path: str):
    """Check if given path has write access."""
    return os.access(path, os.W_OK)


def prepend_with_env_path(config_key, *paths):
    """Prepend given paths with base path specified in env_config.yml for given config_key"""
    # Local import to avoid circular dependency
    from e2e_tests.test_utils.env_tools import Environment
    return Environment.abs_path(config_key, *paths)


def search_model_path_recursively(config_key, model_name):
    from e2e_tests.test_utils.env_tools import Environment
    search_pattern = Environment.abs_path(config_key) + '/**/' + model_name
    path_found = list(iglob(search_pattern, recursive=True))
    if len(path_found) == 1:
        return path_found[0]
    elif len(path_found) == 0:
        raise FileNotFoundError("File not found for pattern {}".format(search_pattern))
    else:
        raise ValueError("More than one file with {} name".format(model_name))


def proto_from_model(caffemodel):
    """Construct .prototxt path from model.caffemodel path."""
    return str(PurePath(caffemodel).with_suffix(".prototxt"))


def ref_from_model(model_name, framework, opset="", check_empty_ref_path=True, extension=".npz"):
    """Construct reference path from model base name."""
    ref_filename = os.path.splitext(os.path.basename(model_name))[
                       0] + extension  # split is needed in case filename contains . symbol
    ref_path = prepend_with_env_path("references", framework, opset, ref_filename)
    if check_empty_ref_path and not os.path.isfile(ref_path):
        ref_path = prepend_with_env_path("references_repo", framework, opset, ref_filename)
    return ref_path


def symbol_from_model(mxnetmodel):
    """Construct symbolic graph path from mxnet model path."""
    # If mxnet model contains -NNNN patter (epochs number) it will be stripped
    if re.search(r"(-[0-9]{4})", mxnetmodel):
        return os.path.splitext(mxnetmodel)[0][:-5] + '-symbol.json'
    else:
        return os.path.splitext(mxnetmodel)[0] + '-symbol.json'


def md5(file_path):
    hash_md5 = hashlib.md5()
    if not os.path.exists(file_path):
        return None
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


class DirLockingHandler:

    def __init__(self, target_dir):
        self.target_dir = resolve_dir_path(target_dir, as_str=True)
        self.fallback_dir = os.getcwd()
        self.writable = is_writable(self.target_dir)
        self._lock_file = Path(self.target_dir) / '.lock'

    def is_locked(self):
        return self._lock_file.exists()

    def lock(self):
        # Local import to avoid cyclic import
        from e2e_tests.test_utils.env_tools import Environment
        if self.writable:
            if not self.is_locked():
                log.info("Marking {} directory as locked".format(self.target_dir))
                self._lock_file.touch(exist_ok=False)
                Environment.locked_dirs.append(self.target_dir)
                lock_info = "Locked at {} by host {} process PID {} running under {}".format(datetime.datetime.now(),
                                                                                             socket.gethostname(),
                                                                                             os.getpid(),
                                                                                             getpass.getuser())
                self._lock_file.write_text(lock_info)
        else:
            raise PermissionError(
                "Failed to lock target directory {} because it's not writable!".format(self.target_dir))

    def unlock(self):
        # Local import to avoid cyclic import
        from e2e_tests.test_utils.env_tools import Environment
        if self.is_locked():
            self._lock_file.unlink()
            if self.target_dir in Environment.locked_dirs:
                Environment.locked_dirs.remove(self.target_dir)
            log.info("Marking {} directory as unlocked".format(self.target_dir))
        else:
            log.warning("Target directory {} is not locked".format(self.target_dir))

    def execute_after_unlock(self, max_wait_time: int = 600,
                             exec_after_unlock: callable = lambda *args, **kwargs: log.info("Directory unlocked"),
                             fallback_to_cwd=True,
                             *args,
                             **kwargs):
        wait_iters = math.ceil(max_wait_time / 30)
        if self.is_locked():
            log.info("Target directory {} locked".format(self.target_dir))
        for i in range(wait_iters):
            if self.is_locked():
                log.info("[{}] Waiting for directory unlocking".format(i + 1))
                time.sleep(30)
            else:
                self.lock()
                try:
                    exec_after_unlock(*args, **kwargs)
                except Exception as e:
                    log.error(str(e))
                finally:
                    self.unlock()
                    break
        else:
            if self.is_locked():
                if not fallback_to_cwd:
                    raise TimeoutError(
                        "Timeout exceeded. Directory {} was not unlocked after {} seconds.".format(self.target_dir,
                                                                                                   max_wait_time))
                else:
                    # TODO: think about fallback latter
                    pass


def get_abs_path(entry: Union[str, Path]) -> Path:
    """ Return pathlib.Path object representing absolute path for the entry """
    try:
        path = Path(entry).expanduser().absolute()
    except TypeError as type_error:
        raise TypeError(f'"{entry}" is expected to be a path-like') from type_error
    return path


def get_rel_path(entry: Union[str, Path], start_path: Union[str, Path]) -> Path:
    """ Return pathlib.Path object representing path for the entry relative to start_path """
    return Path(entry).resolve().relative_to(Path(start_path).resolve())


def get_dir_path(entry: Union[str, Path]) -> Path:
    """Return pathlib.Path object representing
    - absolute path for the entry if entry is directory,
    - absolute path for the entry.parent if entry is file
    """
    path = get_abs_path(entry)
    return path if path.is_dir() else path.parent


def get_ir(search_dir: Path, model_name: str) -> dict:
    """Look for IR (xml/bin files) with specified model_name in specified search_dir, return dict
    with absolute paths to IR components if exist or empty dict otherwise, for example:
    { model: <search_dir>/<model_name>.xml, weights: <search_dir>/<model_name>.bin }
    """
    ir = {}

    filename_pattern = model_name or "*"
    models_list = list(search_dir.glob(f"{filename_pattern}.xml"))
    if models_list:
        model = get_abs_path(models_list[0])
        weights = model.with_suffix(".bin")
        if weights.exists():
            ir = {"model": model, "weights": weights}

    return ir
