# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class TopKExtender(Extender):
    op = 'TopK'

    @staticmethod
    def extend(op: Node):
        if op.out_port(0).disconnected():
            op['remove_values_output'] = True
        if op.has_valid('index_element_type'):
            op['index_element_type'] = destination_type_to_np_data_type(op.index_element_type)
