# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import abc
from collections import defaultdict
from typing import Dict

import numpy as np

from openvino.tools.mo.front.common.partial_infer.utils import int64_array
from openvino.tools.mo.middle.passes.convert_data_type import np_data_type_to_destination_type


class RTInfo:
    """
    Class that stores runtime information.
    """

    def __init__(self):
        """
        Dictionary with runtime information.
        Key is a tuple that contains name of runtime info attribute and version of the attribute.
        Value is an instance of a class derived from RTInfoElement that represents a particular runtime info attribute.

        Example of usage:
        rt_info = RTInfo()
        rt_info.info[('old_api_map_order', 0)] = OldAPIMapOrder()

        """
        self.info = defaultdict(dict)

    def contains(self, attribute_name: str):
        attr_count = [key[0] for key in list(self.info.keys())].count(attribute_name)
        assert attr_count <= 1, 'Incorrect rt_info attribute, got more than one {}.'.format(attribute_name)
        return attr_count > 0

    def get_attribute_version(self, attribute_name: str):
        for name, version in list(self.info.keys()):
            if name == attribute_name:
                return version
        raise Exception("rt_info does not contain attribute with name {}".format(attribute_name))


class RTInfoElement:
    """
    Class that stores element of runtime information.
    """

    @abc.abstractmethod
    def serialize(self, node) -> Dict:
        """
        Serialize method for RTInfoElement.
        """

    @abc.abstractmethod
    def get_version(self):
        """
        Get version of RTInfoElement.
        """

    @abc.abstractmethod
    def get_name(self):
        """
        Get name of RTInfoElement.
        """


class OldAPIMapOrder(RTInfoElement):
    """
    Class that stores transpose order required for obtaining IR in old API.
    """

    def __init__(self, version=0):
        self.info = defaultdict(dict)
        self.version = version
        self.name = "old_api_map_order"

    def old_api_transpose_parameter(self, inv: int64_array):
        self.info['inverse_order'] = inv

    def old_api_transpose_result(self, order: int64_array):
        self.info['order'] = order

    def serialize_old_api_map_for_parameter(self, node) -> Dict:
        if 'inverse_order' not in self.info:
            return {}
        return {'value': ','.join(map(str, self.info['inverse_order']))}

    def serialize_old_api_map_for_result(self, node) -> Dict:
        if 'order' not in self.info:
            return {}
        return {'value': ','.join(map(str, self.info['order']))}

    def serialize(self, node) -> Dict:
        result = {}
        if node.soft_get('type') == 'Parameter':
            result = self.serialize_old_api_map_for_parameter(node)
        elif node.soft_get('type') == 'Result':
            result = self.serialize_old_api_map_for_result(node)
        return result

    def get_version(self):
        return self.version

    def get_name(self):
        return self.name


class OldAPIMapElementType(RTInfoElement):
    """
    Class that stores legacy type required for obtaining IR in old API.
    """
    def __init__(self, version=0):
        self.info = defaultdict(dict)
        self.version = version
        self.name = "old_api_map_element_type"

    def set_legacy_type(self, legacy_type: np.dtype):
        self.info['legacy_type'] = legacy_type

    def serialize(self, node) -> Dict:
        if 'legacy_type' not in self.info:
            return {}
        return {'value': np_data_type_to_destination_type(self.info['legacy_type'])}

    def get_version(self):
        return self.version

    def get_name(self):
        return self.name
