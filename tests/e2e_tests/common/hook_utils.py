# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from _pytest.python import Function

from . import config
from .config import bug_ids, components_ids, req_ids
from .logger import get_logger
from .marks import MarkGeneral, MarkRunType

logger = get_logger(__name__)
_current_test_run = ""
test_run_reporters = {}


def get_required_marker_ids_for_test_run():
    required_marker_ids = []
    if bug_ids is not None:
        required_marker_ids.append(bug_ids)
    if req_ids is not None:
        required_marker_ids.append(req_ids)
    if components_ids is not None:
        required_marker_ids.append(components_ids)
    if len(required_marker_ids) == 0:
        return None
    return required_marker_ids


def update_components(item):
    components = item.get_closest_marker(MarkGeneral.COMPONENTS.mark)
    if components is not None:
        current_components = next(
            (component for component in item.own_markers if component.name == MarkGeneral.COMPONENTS.mark), None)
        if current_components is None:
            item.own_markers.append(components)


def update_markers(item, test_type, markers, marker_type):
    marker = item.get_closest_marker(marker_type)
    if marker is not None:
        if test_type not in markers:
            markers[test_type] = set()
        markers[test_type].update(set(marker.args))


def deselect_items(items, config, deselected):
    config.hook.pytest_deselected(items=deselected)
    for item in deselected:
        test_name = item.parent.nodeid
        # nodeid comes in a way:
        # 1) test.py::TestClass::()
        # 2) test.py::
        if test_name[-2:] == "()":
            test_name = test_name[:-2]
        else:
            test_name += "::"

        test_name += item.name
        logger.info("Deselecting test: " + test_name)
        items.remove(item)


def deselect(item, test_type, required_marker_ids):
    if isinstance(item, Function):
        if test_type is None:
            logger.warning(f"Test type for item={item} is None")
            return True
        if required_marker_ids is not None:
            for marker_id in required_marker_ids:
                if _is_test_marker_id_is_matched_with_id(item, marker_id):
                    return False
            return True
        else:
            if _test_deselected(item):
                return True
    return False


def _test_deselected(item):
    result = any([
        MarkRunType.get_test_type_mark(item) == MarkRunType.TEST_MARK_ON_COMMIT and not config.run_on_commit_tests,
        MarkRunType.get_test_type_mark(item) == MarkRunType.TEST_MARK_REGRESSION and not config.run_regression_tests,
        MarkRunType.get_test_type_mark(item) == MarkRunType.TEST_MARK_ENABLING and not config.run_enabling_tests,
    ])
    return result


def _is_test_marker_id_is_matched_with_id(test, id_to_check: str):
    for marker in test.own_markers:
        if marker.name is MarkGeneral.BUGS.value or marker.name is MarkGeneral.REQIDS.value or \
                marker.name is MarkGeneral.COMPONENTS.value:
            marker_arg = marker.args[0]
            if isinstance(marker_arg, dict):
                for param in marker_arg:
                    if param is None:
                        if id_to_check in str(marker_arg.values):
                            return True
                    else:
                        if param in test.name:
                            if id_to_check in str(marker_arg.values()):
                                return True
            elif isinstance(marker_arg, str):
                if id_to_check in marker_arg:
                    return True
            else:
                raise RuntimeError(f"Test {test.name} do not have mark in correct form. Form: {type(marker_arg)} ")
    return False


def _get_current_test_run():
    return _current_test_run


def _set_current_test_run(test_run):
    _current_test_run = test_run
    return _current_test_run
