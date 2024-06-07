# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import json

from openvino.tools.mo.front.common.replacement import FrontReplacementPattern
from openvino.tools.mo.graph.graph import Graph
from openvino.tools.mo.utils.custom_replacement_config import parse_custom_replacement_config_file
from openvino.tools.mo.utils.error import Error
from openvino.tools.mo.utils.utils import refer_to_faq_msg


class TensorflowCustomOperationsConfigUpdate(FrontReplacementPattern):
    enabled = True
    graph_condition = [lambda graph: graph.graph['cmd_params'].tensorflow_custom_operations_config_update is not None]

    def run_before(self):
        return []

    def run_after(self):
        from openvino.tools.mo.front.freeze_placeholder_value import FreezePlaceholderValue
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
