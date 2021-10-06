# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import abc
from collections import defaultdict
from typing import Dict

import numpy as np

from mo.middle.passes.convert_data_type import np_data_type_to_destination_type


class RTInfo:
    """
    Class that stores runtime information.
    """

    def __init__(self):
        """
        Dictionary with runtime information.
        Key is a tuple that contains name of runtime info attribute and version version of the attribute.
        Value is an instance of a class derived from RTInfoElement that represents a particular runtime info attribute.

        Example of usage:
        rt_info = RTInfo()
        rt_info.info[('old_api_map', 0)] = OldAPIMap()

        """
        self.info = defaultdict(dict)


class RTInfoElement:
    """
    Class that stores element of runtime information.
    """

    @abc.abstractmethod
    def serialize(self, node) -> Dict:
        """
        Serialize method for RTInfoElement.
        """


class OldAPIMap(RTInfoElement):
    """
    Class that stores old API map information, which includes legacy type and transpose orders
    required for obtaining IR in old API.
    """

    def __init__(self):
        self.info = defaultdict(dict)

    def old_api_transpose_parameter(self, inv: np.array):
        self.info['inverse_order'] = inv

    def old_api_transpose_result(self, order: np.array):
        self.info['order'] = order

    def old_api_convert(self, legacy_type: np.dtype):
        self.info['legacy_type'] = legacy_type

    def serialize_old_api_map_for_parameter(self) -> Dict:
        if 'legacy_type' not in self.info and 'inverse_order' not in self.info:
            return {}
        result = {'order': '', 'element_type': 'undefined'}
        if 'legacy_type' in self.info:
            result['element_type'] = np_data_type_to_destination_type(self.info['legacy_type'])

        if 'inverse_order' in self.info:
            result['order'] = ','.join(map(str, self.info['inverse_order']))
        return result

    def serialize_old_api_map_for_result(self) -> Dict:
        if 'order' in self.info:
            return {'order': ','.join(map(str, self.info['order'])), 'element_type': 'undefined'}
        return {}

    def serialize(self, node) -> Dict:
        result = {}
        if node.soft_get('type') == 'Parameter':
            result = self.serialize_old_api_map_for_parameter()
        elif node.soft_get('type') == 'Result':
            result = self.serialize_old_api_map_for_result()
        return result
