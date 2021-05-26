"""Pytest utility functions."""
# pylint:disable=import-error
import pytest
from multiprocessing import TimeoutError
from collections import namedtuple
from _pytest.mark import Mark, MarkDecorator

"""Mark generator to specify pytest marks in tests in common format
:param target_runner: name of the runner for which required to specify pytest mark (e.g. "test_run")
:param pytest_mark: pytest mark (e.g. "skip", "xfail") or any another mark (e.g. "onnx")
:param is_simple_mark: bool value to split pytest marks and another marks
"""
mark = namedtuple("mark", ("pytest_mark", "target_runner", "is_simple_mark"))
# default values for "target_runner" and "is_simple_mark" fields respectively
mark.__new__.__defaults__ = ("all", False)


class XFailMarkWrapper(Mark):
    def __init__(self, regexps: list, match_mode: str = "any", *args, **kwargs):
        """
        Class constructs 'xfail'-like mark with additional fields
        :param regexps: regexp to search in test logs
        :param match_mode: 'any' or
        :param args:
        :param kwargs:
        """
        super().__init__('xfail', *args, **kwargs)
        object.__setattr__(self, "regexps", regexps)
        object.__setattr__(self, "match_mode", match_mode)


def skip(reason):
    """Generate skip marker.

    :param reason: reason why marker is generated

    :return: pytest marker
    """
    return pytest.mark.skip(True, reason=reason), reason


def xfail(reason, regexps="", match_mode="any"):
    """Generate xfail marker.

    :param reason: reason why marker is generated
    :param regexps: list of regular expressions for matching xfail reason on test's status
    :param match_mode: defines that "all" or "any" specified regexps should be matched

    :return: pytest marker
    """
    regexps = [regexps] if not isinstance(regexps, list) else regexps
    mark = XFailMarkWrapper(regexps=regexps, match_mode=match_mode,
                            args=(True,), kwargs={"reason": reason, "strict": True})
    return MarkDecorator(mark=mark), reason


def timeout(seconds, reason):
    """Generate timeout marker.

    :param seconds: number of seconds until timeout is reached

    :param reason: reason why marker is generated

    :return: pytest marker
    """
    return pytest.mark.timeout(seconds), reason


def warning(reason):
    """Generate warning marker.

    :param reason: reason why marker is generated

    :return: pytest marker
    """
    return pytest.mark.warning(reason), reason
