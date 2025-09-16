#!/usr/bin/env python

#
# Copyright (C) 2025 Intel Corporation
# SPDX-License-Identifier: Apache 2.0
#

from common.converters import shape_to_list, layout_to_str
from common.enums import InputSourceFileType
from schema.validator import JSONSchemaValidator

class InputSource(JSONSchemaValidator):
    schema = JSONSchemaValidator.load_from_file("input_source")

    def __canonize_image_data__(self):
        if "convert" in self.keys():
            if "shape" in self["convert"].keys():
                self["convert"]["shape"] = shape_to_list(self["convert"]["shape"])
            if "layout" in self["convert"].keys():
                self["convert"]["layout"] = layout_to_str(self["convert"]["layout"])

    def __canonize_binary_data__(self):
        if "shape" in self.keys():
            self["shape"] = shape_to_list(self["shape"])
        if "layout" in self.keys():
            self["layout"] = layout_to_str(self["layout"])

    def __canonize_data__(self):
        if "type" not in self.keys():
            self["type"] = InputSourceFileType.bin.name

        canonizer = {InputSourceFileType.image.name : InputSource.__canonize_image_data__,
                    InputSourceFileType.bin.name : InputSource.__canonize_binary_data__}
        canonizer[self["type"]](self)

    def __init__(self, input_json, *args, **kwargs):
        super().__init__(InputSource.schema, input_json, *args, **kwargs)
        self.__canonize_data__()
