# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Local pytest plugins shared code."""
import importlib.util
import inspect
import os
from fnmatch import fnmatch
from glob import glob
from inspect import getsourcefile

from _pytest.mark import MarkDecorator

from e2e_tests.common.pytest_utils import mark as Mark

from e2e_tests.test_utils.test_utils import BrokenTestException


def apply_glob(paths, file_ext="py"):
    """
    Apply glob to paths list.

    If path is file and matches pattern *.<file_ext>, add it.

    If path is directory, search for pattern <path>/**/*.<file_ext> recursively.

    If path contains special characters (*, ?, [, ], !),
    pass path to glob and add resolved values that match *.<file_ext>.

    :param paths:   list of paths
    :param file_ext:    file extension to filter by (i.e. if "py", only .py
                        files are returned)
    :return:    resolved paths
    """
    file_pattern = '*.{ext}'.format(ext=file_ext)
    globbed_paths = []
    for path in paths:
        # resolve files
        if os.path.isfile(path):
            if fnmatch(path, file_pattern):
                globbed_paths.append(path)
        # resolve directories
        elif os.path.isdir(path):
            globbed_paths.extend(
                glob(
                    '{dir}/**/{file}'.format(dir=path, file=file_pattern),
                    recursive=True))
        # resolve patterns
        elif any(special in path for special in ['*', '?', '[', ']', '!']):
            resolved = glob(path, recursive=True)
            globbed_paths.extend(
                [entry for entry in resolved if fnmatch(entry, file_pattern)])
    return list(set(globbed_paths))


def find_tests(modules, attributes):
    """
    Find tests given list of modules where to look for.

    If class has all attributes specified, append it to found tests.

    :param modules: .py files with test classes
    :param attributes:  class attributes that each test class must have
    :return:    found test classes
    """
    modules = apply_glob(modules)
    tests = []
    broken_modules = []

    for module in modules:
        name = os.path.splitext(os.path.basename(module))[0]
        spec = importlib.util.spec_from_file_location(name, module)
        try:
            loaded_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(loaded_module)
        except Exception as e:
            broken_modules.append((module, str(e)))
            continue
        classes = inspect.getmembers(loaded_module, predicate=inspect.isclass)
        for cls in classes:
            if all(getattr(cls[1], attr, False) for attr in attributes):
                setattr(cls[1], "definition_path", module)
                tests.append(cls[1])

    return tests, broken_modules


def set_pytest_marks(_test, _object, _runner, log):
    """ Set pytest markers from object to the test according to test runner. """
    _err = False
    if hasattr(_object, '__pytest_marks__'):
        for mark in _object.__pytest_marks__:
            if isinstance(mark, MarkDecorator):
                _test.add_marker(mark)
                continue
            if not isinstance(mark, Mark):
                _err = True
                log.error("Current mark '{}' for instance '{}' from '{}' isn't wrapped in 'mark' from '{}'"
                          .format(mark, str(_object), _object.definition_path, getsourcefile(Mark)))
                continue
            if mark.target_runner != "all" and mark.target_runner != _runner:
                continue
            if mark.is_simple_mark:
                mark_to_add = str(mark.pytest_mark)
            else:
                try:
                    mark_to_add, reason = mark.pytest_mark
                except ValueError as ve:
                    _err = True
                    log.exception("Error with marks for {}".format(str(_object)), exc_info=ve)
                    continue
                if mark_to_add is None:  # skip None values
                    continue
                if not reason:
                    _err = True
                    log.error("Mark '{mark}' exists in instance '{instance}' without specified reason"
                              .format(mark=mark_to_add, instance=str(_object)))
                    continue
            _test.add_marker(mark_to_add)
    if _err:
        raise BrokenTestException
