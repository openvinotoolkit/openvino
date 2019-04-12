"""
 Copyright (c) 2019 Intel Corporation

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
import json

from mo.front.common.replacement import FrontReplacementPattern
from mo.graph.graph import Graph
from mo.utils.custom_replacement_config import parse_custom_replacement_config_file
from mo.utils.error import Error
from mo.utils.utils import refer_to_faq_msg


class TensorflowCustomOperationsConfigUpdate(FrontReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].tensorflow_custom_operations_config_update is not None]

    def run_before(self):
        return []

    def run_after(self):
        from extensions.front.freeze_placeholder_value import FreezePlaceholderValue
        return [FreezePlaceholderValue]

    @staticmethod
    def save_custom_replacement_config_file(descriptions: list, file_name: str):
        """
        Save custom layer(s) description(s) to the file.
        :param file_name: file to save description information to.
        :param descriptions: list with instances of the CustomLayerDescriptor classes.
        :return: True if operation is successful.
        """
        try:
            json.dump([replacement_desc.get_config_file_representation() for replacement_desc in descriptions],
                      open(file_name, "w"), indent=4, sort_keys=True)
        except Exception as ex:
            raise Error("failed to update configuration file {}: {}".format(file_name, str(ex)))

    def find_and_replace_pattern(self, graph: Graph):
        argv = graph.graph['cmd_params']
        file_name = argv.tensorflow_custom_operations_config_update

        data = parse_custom_replacement_config_file(file_name)
        if data is None:
            raise Error("Cannot update the file '{}' because it is broken. ".format(file_name) + refer_to_faq_msg(73))

        for replacement_desc in data:
            replacement_desc.update_custom_replacement_attributes(graph)

        self.save_custom_replacement_config_file(data, file_name)
