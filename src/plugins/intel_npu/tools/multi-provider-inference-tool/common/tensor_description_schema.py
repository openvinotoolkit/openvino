#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

from common.converters import shape_to_list, layout_to_str
from schema.validator import JSONSchemaValidator


class TensorInfoData(JSONSchemaValidator):
    schema = JSONSchemaValidator.load_from_file("tensor")

    def __canonize_data__(self):
        if "shape" in self.keys():
            self["shape"] = shape_to_list(self["shape"])
        if "layout" in self.keys():
            self["layout"] = layout_to_str(self["layout"])

    def __init__(self, input_json, *args, **kwargs):
        super().__init__(TensorInfoData.schema, input_json, *args, **kwargs)
        self.__canonize_data__()
