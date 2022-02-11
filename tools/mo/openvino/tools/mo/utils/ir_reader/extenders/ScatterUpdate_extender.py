# Copyright (C) 2018-2022 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.middle.passes.convert_data_type import destination_type_to_np_data_type

from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender


class ScatterUpdate(Extender):
    op = 'ScatterUpdate'

    @staticmethod
    def extend(op: Node):
        op['infer'] = Extender.use_shapes_from_ir
