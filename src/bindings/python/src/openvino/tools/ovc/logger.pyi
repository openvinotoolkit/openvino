# type: ignore
from copy import copy
from __future__ import annotations
import importlib as importlib
import logging
import logging as log
import os as os
import re as re
import typing
__all__ = ['LvlFormatter', 'TagFilter', 'copy', 'get_logger_state', 'handler_num', 'importlib', 'init_logger', 'log', 'os', 're', 'restore_logger_state']
class LvlFormatter(logging.Formatter):
    format_dict: typing.ClassVar[dict] = {10: '[ %(asctime)s ] [ %(levelname)s ] [ %(module)s:%(lineno)d ]  %(msg)s', 20: '[ %(levelname)s ]  %(msg)s', 30: '[ WARNING ]  %(msg)s', 40: '[ %(levelname)s ]  %(msg)s', 50: '[ %(levelname)s ]  %(msg)s', 'framework_error': '[ FRAMEWORK ERROR ]  %(msg)s', 'analysis_info': '[ ANALYSIS INFO ]  %(msg)s'}
    def __init__(self, lvl, fmt = None):
        ...
    def format(self, record: logging.LogRecord):
        ...
class TagFilter(logging.Filter):
    def __init__(self, regex: str):
        ...
    def filter(self, record: logging.LogRecord):
        ...
def get_logger_state():
    ...
def init_logger(lvl: str, verbose: bool, python_api_used: bool):
    ...
def restore_logger_state(state: tuple):
    ...
handler_num: int = 0
