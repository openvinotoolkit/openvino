# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import logging as log
import os
import re
import sys
from argparse import Namespace
from copy import copy

# WA for abseil bug that affects logging while importing TF starting 1.14 version
# Link to original issue: https://github.com/abseil/abseil-py/issues/99
if importlib.util.find_spec('absl') is not None:
    import absl.logging

    log.root.removeHandler(absl.logging._absl_handler) # pylint: disable=c-extension-no-member

handler_num = 0


class LvlFormatter(log.Formatter):
    format_dict = {
        log.DEBUG: "[ %(asctime)s ] [ %(levelname)s ] [ %(module)s:%(lineno)d ]  %(msg)s",
        log.INFO: "[ %(levelname)s ]  %(msg)s",
        log.WARNING: "[ WARNING ]  %(msg)s",
        log.ERROR: "[ %(levelname)s ]  %(msg)s",
        log.CRITICAL: "[ %(levelname)s ]  %(msg)s",
        'framework_error': "[ FRAMEWORK ERROR ]  %(msg)s",
        'analysis_info': "[ ANALYSIS INFO ]  %(msg)s"
    }

    def __init__(self, lvl, fmt=None):
        log.Formatter.__init__(self, fmt)
        self.lvl = lvl

    def format(self, record: log.LogRecord):
        if self.lvl == 'DEBUG':
            self._style._fmt = self.format_dict[log.DEBUG]
        else:
            self._style._fmt = self.format_dict[record.levelno]
        if 'is_warning' in record.__dict__.keys():
            self._style._fmt = self.format_dict[log.WARNING]
        if 'framework_error' in record.__dict__.keys():
            self._style._fmt = self.format_dict['framework_error']
        if 'analysis_info' in record.__dict__.keys():
            self._style._fmt = self.format_dict['analysis_info']
        return log.Formatter.format(self, record)


class TagFilter(log.Filter):
    def __init__(self, regex: str):
        self.regex = regex

    def filter(self, record: log.LogRecord):
        if record.__dict__['funcName'] == 'load_grammar':  # for nx not to log into our logs
            return False
        if self.regex:
            if 'tag' in record.__dict__.keys():
                tag = record.__dict__['tag']
                return re.findall(self.regex, tag)
            else:
                return False
        return True  # if regex wasn't set print all logs


def init_logger(lvl: str, silent: bool):
    global handler_num
    log_exp = os.environ.get('MO_LOG_PATTERN')
    if silent:
        lvl = 'ERROR'
    fmt = LvlFormatter(lvl=lvl)
    handler = log.StreamHandler()
    handler.setFormatter(fmt)
    logger = log.getLogger()
    logger.setLevel(lvl)
    logger.addFilter(TagFilter(regex=log_exp))
    if handler_num == 0 and len(logger.handlers) == 0:
        logger.addHandler(handler)
        handler_num += 1

def get_logger_state():
    logger = log.getLogger()
    return logger.level, copy(logger.filters), copy(logger.handlers)

def restore_logger_state(state: tuple):
    level, filters, handlers = state
    logger = log.getLogger()
    logger.setLevel(level)
    logger.filters = filters
    logger.handlers = handlers


def progress_bar(function: callable):
    """
    Decorator for model conversion pipeline progress display
    Works in combination with function: mo.utils.class_registration.apply_transform
    """

    def wrapper(*args, **kwargs):
        for arg in ['graph', 'curr_transform_num', 'num_transforms']:
            msg = 'Progress bar decorator is enabled for Model Conversion API transformation applying cycle only. ' \
                  'Argument `{}` {}'

            assert arg in kwargs, msg.format(arg, 'is missing')
            assert kwargs[arg] is not None, msg.format(arg, 'should not be None')

        if 'progress' in kwargs['graph'].graph['cmd_params'] and kwargs['graph'].graph['cmd_params'].progress:
            bar_len = 20
            total_replacers_count = kwargs['num_transforms']

            def progress(i):
                return int((i + 1) / total_replacers_count * bar_len)

            def percent(i):
                return (i + 1) / total_replacers_count * 100

            end = '' if not kwargs['graph'].graph['cmd_params'].stream_output else '\n'
            curr_i = kwargs['curr_transform_num']
            print('\rProgress: [{:{}}]{:>7.2f}% done'.format('.' * progress(curr_i), bar_len, percent(curr_i)), end=end)

            sys.stdout.flush()

        function(*args, **kwargs)

    return wrapper

def progress_printer(argv: Namespace):
    """
    A higher-order factory function returning a configurable callback displaying a progress bar
    Depending on the configuration stored in 'argv' the progress bar can be one-line, multi-line, or silent.
    """
    def _progress_bar(progress, total, completed, endline):
        bar_len = 20

        def dots():
            return '.' * int(progress * bar_len)

        print('\rProgress: [{:{}}]{:>7.2f}% done'.format(dots(), bar_len, progress*100), end=endline)
        sys.stdout.flush()

    def no_progress_bar(progress, total, completed):
        """ A 'dummy' progressbar which doesn't print anything """
        pass

    def oneline_progress_bar(progress, total, completed):
        """ A callback that always prints the progress in the same line (mimics real GUI progress bar)"""
        _progress_bar(progress, total, completed, '')

    def newline_progress_bar(progress, total, completed):
        """ A callback that prints an updated progress bar in separate lines """
        _progress_bar(progress, total, completed, '\n')

    if "progress" in argv and argv.progress:
        if "stream_output" in argv and argv.stream_output:
            return newline_progress_bar
        else:
            return oneline_progress_bar
    else:
        return no_progress_bar
