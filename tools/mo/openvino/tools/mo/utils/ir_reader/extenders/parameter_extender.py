# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino.tools.mo.front.common.partial_infer.utils import int64_array, shape_array, dynamic_dimension_value
from openvino.tools.mo.middle.passes.convert_data_type import destination_type_to_np_data_type
from openvino.tools.mo.utils.graph import Node
from openvino.tools.mo.utils.ir_reader.extender import Extender
from openvino.runtime import PartialShape, Dimension


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

            # Remove brackets from shape splited by comma separator
            if isinstance(op.shape[0], str) and op.shape[0][0] == '[':
                op.shape[0] = op.shape[0][1:]
            if isinstance(op.shape[-1], str) and op.shape[-1][-1] == ']':
                op.shape[-1] = op.shape[-1][:-1]

            shape = op.shape.copy()
            has_shapes_with_boundaries = False
            for i, dim in enumerate(op.shape):
                if dim == -1 or (isinstance(dim, str) and ".." in dim):
                    shape[i] = -1
                    # Check only if dim is not int
                    if not isinstance(dim, int) and '..' in dim:
                        has_shapes_with_boundaries = True
            shape = shape_array([int(d) if d not in [-1, '?'] else dynamic_dimension_value for d in shape])

            if has_shapes_with_boundaries:
                shape_list = []
                for dim in op.shape:
                    shape_list.append(Dimension(dim))

                # This value is used only for serialization of partial shapes with boundaries
                # for Parameter node.
                # 'user_shape' is not used in shape inference, as propagation of partial shapes with boundaries
                # is not implemented in MO.
                op['user_shape'] = PartialShape(shape_list)

            # If 'user_shape' is not set, 'shape' attribute is used for serialization.
            # 'shape' is also used for shape inference.
            op.shape = shape
