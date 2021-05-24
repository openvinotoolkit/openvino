# Copyright (C) 2018-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from mo.front.common.partial_infer.utils import int64_array
from mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from mo.utils.graph import Node
from mo.utils.ir_reader.extender import Extender


class Parameter_extender(Extender):
    op = 'Parameter'

    @staticmethod
    def extend(op: Node):
        assert op.has_valid('element_type'), 'Parameter node {} has missed element_type attr!'.format(op.name)
        op['data_type'] = destination_type_to_np_data_type(op.element_type)
        if op.shape == '':
            op.shape = int64_array([])
        else:
            Extender.attr_to_list(op, 'shape')
