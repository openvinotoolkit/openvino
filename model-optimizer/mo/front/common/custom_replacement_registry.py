"""
 Copyright (c) 2017-2019 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import logging as log
import os

from mo.utils.custom_replacement_config import parse_custom_replacement_config_file
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


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
