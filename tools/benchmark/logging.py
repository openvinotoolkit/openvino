"""
Copyright (C) 2018-2019 Intel Corporation

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

import logging
import logging.config
import sys
import warnings

# TODO: move to utils
_DEFAULT_LOGGER_NAME = 'openvino.tools.benchmark'
_DEFAULT_LOG_FILE = 'openvino.tools.benchmark.log'

PRINT_INFO = logging.INFO + 5
logging.addLevelName(PRINT_INFO, "PRINT_INFO")

_LOG_LEVEL_ENVIRON = "CALIBRATION_TOOL_LOG_LEVEL"
# _LOGGING_LEVEL = logging.getLevelName(os.environ.get(_LOG_LEVEL_ENVIRON, PRINT_INFO))
# TODO: refactoring: remove, use original line
_LOGGING_LEVEL = "DEBUG"


class LoggingFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord):
        if record.levelno == PRINT_INFO:
            return record.msg
        return super().format(record)


class ConsoleHandler(logging.StreamHandler):
    def __init__(self, default_stream=sys.stdout):
        super().__init__(default_stream)
        self.default_stream = default_stream
        self.err_stream = sys.stderr

    def emit(self, record):
        if record.levelno >= logging.WARNING:
            self.stream = self.err_stream
        else:
            self.stream = self.default_stream
        super().emit(record)


_LOGGING_CONFIGURATION = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            '()': LoggingFormatter,
            'format': '%(asctime)s %(name)s %(levelname)s: %(message)s',
            'datefmt': '%H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s %(name)s %(levelname)s: %(message)s'
        }
    },
    'handlers': {
        'console': {
            'level': 'DEBUG',
            '()': ConsoleHandler,
            'formatter': 'default',
        }
    },

    'loggers': {
        _DEFAULT_LOGGER_NAME: {
            'handlers': ['console'],
            'level': _LOGGING_LEVEL,
            'propagate': False
        }
    }
}

logging.config.dictConfig(_LOGGING_CONFIGURATION)

_default_logger = logging.getLogger(_DEFAULT_LOGGER_NAME)


def _warning_handler(message, category, filename, lineno):
    s = warnings.formatwarning(message, category, filename, lineno)
    _default_logger.warning(s)


warnings.showwarning = _warning_handler


def get_logger(logger_name: str):
    if logger_name.startswith(_DEFAULT_LOGGER_NAME):
        return _default_logger.getChild(logger_name)
    return logging.getLogger(logger_name)


def error(msg, *args, **kwargs):
    _default_logger.error(msg, *args, **kwargs)


def warning(msg, *args, raise_warning=True, **kwargs):
    if raise_warning:
        warnings.warn(msg)
    else:
        _default_logger.warning(msg, *args, **kwargs)


def info(msg, *args, **kwargs):
    _default_logger.info(msg, *args, **kwargs)


def debug(msg, *args, **kwargs):
    _default_logger.debug(msg, *args, **kwargs)


def print_info(msg, *args, **kwargs):
    _default_logger.log(PRINT_INFO, msg, *args, **kwargs)
