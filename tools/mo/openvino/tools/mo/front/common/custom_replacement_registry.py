# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging as log
import os

from openvino.tools.mo.utils.custom_replacement_config import parse_custom_replacement_config_file
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class CustomReplacementRegistry(object):
    """
    Registry that contains registered custom calls descriptors.
    """

    class __CustomReplacementRegistry:
        def __init__(self):
            self.registry = {}

        def __str__(self):
            return repr(self) + str(self.registry)

    def __init__(self):
        if not CustomReplacementRegistry.instance:
            CustomReplacementRegistry.instance = CustomReplacementRegistry.__CustomReplacementRegistry()
        else:
            pass
            # CustomCallRegistry.instance.val = arg

    def __getattr__(self, name):
        return getattr(self.instance, name)

    instance = None

    def add_custom_replacement_description_from_config(self, file_name: str):
        if not os.path.exists(file_name):
            raise Error("Custom replacement configuration file '{}' doesn't exist. ".format(file_name) +
                        refer_to_faq_msg(46))

        descriptions = parse_custom_replacement_config_file(file_name)
        for desc in descriptions:
            self.registry.setdefault(desc.id, list()).append(desc)
            log.info("Registered custom replacement with id '{}'".format(desc.id))

    def get_custom_replacement_description(self, replacement_id: str):
        if replacement_id in self.registry:
            return self.registry[replacement_id]
        else:
            log.warning("Configuration file for custom replacement with id '{}' doesn't exist".format(replacement_id))
            return None

    def get_all_replacements_descriptions(self):
        result = list()
        for l in self.registry.values():
            result.extend(l)
        return result
