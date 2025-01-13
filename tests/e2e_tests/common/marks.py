# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


import re
from enum import Enum
from itertools import chain
from typing import Union

from _pytest.nodes import Item

from .config import repository_name
from .logger import get_logger

RePattern = type(re.compile(""))
logger = get_logger(__name__)


class MarkMeta(str, Enum):
    def __new__(cls, mark: str, description: str = None, *args):
        obj = str.__new__(cls, mark)  # noqa
        obj._value_ = mark
        obj.description = description
        return obj

    def __init__(self, *args):
        super(MarkMeta, self).__init__()

    def __hash__(self) -> int:
        return hash(self.mark)

    def __format__(self, format_spec):
        return self.mark

    def __repr__(self):
        return self.mark

    def __str__(self):
        return self.mark

    @classmethod
    def get_by_name(cls, name):
        return name

    @property
    def mark(self):
        return self._value_

    @property
    def marker_with_description(self):
        return "{}{}".format(self.mark,
                             ": {}".format(self.description) if self.description is not None else "")

    def __eq__(self, o: object) -> bool:
        if isinstance(o, str):
            return self.mark.__eq__(o)
        return super().__eq__(o)


class ConditionalMark(MarkMeta):
    @classmethod
    def get_conditional_marks_from_item(cls, name, item):
        marks = list(filter(lambda x: x.name == name and x.args is not None, item.keywords.node.own_markers))
        return marks

    @classmethod
    def _test_name_phrase_match_test_item(cls, test_name, item):
        """
            Verify if current 'item' test_name match pytest Mark from test case
        """
        if test_name is None:  # no filtering -> any test_name will match
            return True
        _name = item.keywords.node.originalname
        if isinstance(test_name, RePattern):
            return bool(test_name.match(_name))
        elif isinstance(test_name, str):
            return test_name == _name
        else:
            raise AttributeError(f"Unexpected conditional marker params {test_name} for {item}")

    @classmethod
    def _params_phrase_match_item(cls, params, item):
        """
            Verify if current 'item' parameter match pytest Mark from test case
        """
        if params is None:  # no filtering -> any param will match
            return True
        test_params = item.keywords.node.callspec.id
        if isinstance(params, RePattern):
            return bool(params.match(test_params))
        elif isinstance(params, str):
            return params == test_params
        else:
            raise AttributeError(f"Unexpected conditional marker params {params} for {item}")

    @classmethod
    def _process_single_entry(cls, entry, item):
        """
            Check if mark 'condition' is meet and item parameters match re/str phrase.
            Then return mark value
        """
        value, condition, params, test_name = None, True, None, None
        if isinstance(entry, str):
            # Simple string do not have condition nor parameters.
            value = entry
        elif isinstance(entry, dict):
            value = entry.get('value')  # required
            condition = entry.get('condition', True)
            params = entry.get('params', None)
            test_name = entry.get('test_name', None)
        elif isinstance(entry, tuple):
            value, *_optional = entry
            if isinstance(value, list):
                return cls._process_single_entry(value, item)

            if len(_optional) > 0:
                condition = _optional[0]
            if len(_optional) > 1:
                params = _optional[1]
            if len(_optional) > 2:
                test_name = _optional[2]
        elif isinstance(entry, list):
            for _element in entry:
                value = cls._process_single_entry(_element, item)
                if value:  # Return first match
                    return value
            return None
        else:
            raise AttributeError(f"Unexpected conditional marker entry {entry}")

        if not condition:
            return None

        if not cls._test_name_phrase_match_test_item(test_name, item):
            return None

        return value if cls._params_phrase_match_item(params, item) else None

    @classmethod
    def get_all_marks_values_from_item(cls, item, marks):
        mark_values = []
        for mark in marks:
            values = cls.get_all_marker_values_from_item(item, mark)
            if values:
                mark_values.extend(values)
        return mark_values

    @classmethod
    def get_all_marker_values_from_item(cls, item, mark, _args=None):
        """
            Marker can be set as 'str', 'list', 'tuple', 'dict'.
            Process it accordingly and list of values.
        """
        marker_values = []
        args = _args if _args else mark.args
        if isinstance(args, list):
            for entry in args:
                value = cls._process_single_entry(entry, item)
                if not value:
                    continue
                marker_values.append(value)
        elif isinstance(args, tuple):
            value = cls._process_single_entry(args, item)
            if value:
                marker_values.append(value)
        elif isinstance(args, str):
            marker_values.append(args)
        elif isinstance(args, dict):
            for params, value in args.items():
                if not cls._params_phrase_match_item(params, item):
                    continue
                if isinstance(value, list):
                    marker_values.extend(value)
                else:
                    marker_values.append(value)
        else:
            raise AttributeError(f"Unrecognized conditional marker {mark}")
        return marker_values

    @classmethod
    def get_markers_values_from_item(cls, item, marks):
        result = []
        for mark in marks:
            result.extend(cls.get_all_marker_values_from_item(item, mark))
        return result

    @classmethod
    def get_markers_values_via_conditional_marker(cls, item, name):
        conditional_marks = cls.get_conditional_marks_from_item(name, item)
        markers_values = cls.get_markers_values_from_item(item, conditional_marks)
        return markers_values

    @classmethod
    def get_mark_from_item(cls, item: Item, conditional_marker_name=None):
        marks = cls.get_markers_values_via_conditional_marker(item, conditional_marker_name)
        if not marks:
            return cls.get_closest_mark(item)

        marks = marks[0]
        return marks

    @classmethod
    def get_closest_mark(cls, item: Item):
        for mark in cls:  # type: 'MarkRunType'
            if item.get_closest_marker(mark.mark):
                return mark
        return None

    @classmethod
    def get_by_name(cls, name):
        mark = list(filter(lambda x: x.value == name, list(cls)))
        return mark[0]


class MarkBugs(ConditionalMark):
    @classmethod
    def get_all_bug_marks_values_from_item(cls, item: Item):
        conditional_marks = cls.get_conditional_marks_from_item("bugs", item)
        bugs = cls.get_all_marks_values_from_item(item, conditional_marks)
        return bugs


class MarkGeneral(MarkMeta):
    COMPONENTS = "components"
    REQIDS = "reqids", "Mark requirements tested"


class MarkRunType(ConditionalMark):
    TEST_MARK_COMPONENT = "component", "run component tests", "component"
    TEST_MARK_ON_COMMIT = "api_on_commit", "run api-on-commit tests", "api_on-commit"
    TEST_MARK_REGRESSION = "api_regression", "run api-regression tests", "api_regression"
    TEST_MARK_ENABLING = "api_enabling", "run api-enabling tests", "api_enabling"
    TEST_MARK_MANUAL = "manual", "run api-manual tests", "api_manual"
    TEST_MARK_OTHER = "api_other", "run api-other tests", "api_other"
    TEST_MARK_STRESS_AND_LOAD = "api_stress_and_load", "run api-stress-and-load tests", "api_stress-and-load"
    TEST_MARK_LONG = "api_long", "run api-long tests", "api_long"
    TEST_MARK_PERFORMANCE = "api_performance", "run api-performance tests", "api_performance"

    def __init__(self, mark: str, description: str = None, run_type: str = None) -> None:
        super().__init__(self, mark, description)
        self.run_type = f"{repository_name}_{run_type}" if repository_name is not None else run_type

    @classmethod
    def test_mark_to_test_run_type(cls, test_type_mark: Union['MarkRunType', str]):
        if isinstance(test_type_mark, str):
            return MarkRunType(test_type_mark).run_type
        return test_type_mark.run_type

    @classmethod
    def get_test_type_mark(cls, item: Item):
        mark = cls.get_mark_from_item(item, "test_group")
        if not mark and getattr(item, "parent", None):
            mark = cls.get_mark_from_item(item.parent, "test_group")  # try to deduce test type from parent
        return mark

    @classmethod
    def test_type_mark_to_int(cls, item):
        mark = cls.get_test_type_mark(item)
        if not mark:
            return -1
        return list(cls).index(mark)


class MarksRegistry(tuple):
    MARKERS = "markers"
    MARK_ENUMS = [MarkGeneral, MarkRunType, MarkBugs]

    def __new__(cls) -> 'MarksRegistry':
        # noinspection PyTypeChecker
        return tuple.__new__(cls, [mark for mark in chain(*cls.MARK_ENUMS)])

    @staticmethod
    def register(pytest_config):
        for mark in MarksRegistry():
            pytest_config.addinivalue_line(MarksRegistry.MARKERS, mark.marker_with_description)
