# Copyright (C) 2020-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import sys
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO

from tqdm import tqdm

root = logging.RootLogger(logging.WARNING)
manager = logging.Manager(root)
DEBUG_LEVEL = logging.DEBUG

_is_logging_initialized = False
_progress_bar = None
_stream_output = False
_bar_format = '{desc}{percentage:1.0f}%|{bar}|{elapsed}'


class RegularLogger(logging.Logger):
    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.progress_bar_disabled = True

    def update_progress(self, value):
        pass

    def reset_progress_total(self, total, increase_percent):
        pass

    def info(self, msg, *args, **kwargs):
        if 'force' in kwargs:
            kwargs.pop('force')
        return super().info(msg, *args, **kwargs)


class ProgressBarLogger(RegularLogger):
    stream_output = False

    def __init__(self, name, level=logging.NOTSET):
        super().__init__(name, level)
        self.progress_bar_disabled = False
        if '--stream-output' in sys.argv:
            ProgressBarLogger.stream_output = True

    def update_progress(self, value):
        updating_value = int(value)
        total = _progress_bar.total
        current = _progress_bar.n
        if ProgressBarLogger.stream_output:
            print()
        if updating_value < (total - current):
            _progress_bar.update(updating_value)
            _progress_bar.refresh()
            return
        _progress_bar.n = total
        desc = _progress_bar.desc
        _progress_bar.set_description(desc)
        _progress_bar.refresh()
        _progress_bar.close()

    def reset_progress_total(self, total, increase_percent):
        # pylint: disable=W0603
        global _progress_bar
        desc = _progress_bar.desc
        increase_to = int(total * increase_percent)
        _progress_bar.close()
        _progress_bar = tqdm(range(int(total)), desc=desc, bar_format=_bar_format)
        if ProgressBarLogger.stream_output:
            print()
        _progress_bar.n = increase_to
        _progress_bar.refresh()

    def info(self, msg, *args, **kwargs):
        """Log message with level INFO. Printing to terminal is suppressed.
        :param msg: string to log
        :param force: force message printing to terminal
        """
        if root.getEffectiveLevel() == logging.getLevelName('DEBUG') or kwargs.get('force'):
            _progress_bar.set_description(msg)
        return super().info(msg, *args, **kwargs)


def init_logger(level='INFO', stream=sys.stdout, file_name=None, progress_bar=False):
    # pylint: disable=W0603
    global _is_logging_initialized, _progress_bar
    if _is_logging_initialized:  # avoiding duplication of logging initialization
        return
    _is_logging_initialized = True
    level = logging.getLevelName(level)
    root.setLevel(level)

    if file_name:
        file_handler = logging.FileHandler(file_name)
        file_handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
        root.addHandler(file_handler)

    if progress_bar:
        _progress_bar = tqdm(range(100), leave=False, bar_format=_bar_format)
        return

    handler = logging.StreamHandler(stream=stream)
    handler.setFormatter(logging.Formatter(logging.BASIC_FORMAT))
    root.addHandler(handler)


def get_logger(name):
    if '--progress-bar' in sys.argv:
        manager.setLoggerClass(ProgressBarLogger)
    else:
        manager.setLoggerClass(RegularLogger)
    if name:
        return manager.getLogger(name)
    return root


def stdout_redirect(fn, *args, **kwargs):
    with StringIO() as log_str, redirect_stdout(log_str), redirect_stderr(log_str):
        res = fn(*args, **kwargs)
        if log_str.getvalue():
            get_logger('DEBUG').debug(log_str.getvalue())
        return res
