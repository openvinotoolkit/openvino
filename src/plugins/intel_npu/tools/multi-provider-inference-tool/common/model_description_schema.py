#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from common.converters import shape_to_list, layout_to_str
from schema.validator import JSONSchemaValidator

class ModelInfoData(JSONSchemaValidator):
    schema = JSONSchemaValidator.load_from_file("model")

    def __canonize_data__(self):
        for i, d in self.items():
            if "shape" in d.keys():
                d["shape"] = shape_to_list(d["shape"])
            if "layout" in d.keys():
                d["layout"] = layout_to_str(d["layout"])

    def __init__(self, input_json, *args, **kwargs):
        super().__init__(ModelInfoData.schema, input_json, *args, **kwargs)
        self.__canonize_data__()
