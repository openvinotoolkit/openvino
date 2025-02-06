# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino as ov
from ov_custom_op import Identity

#! [py_frontend_extension_ThresholdedReLU_header]
import openvino.runtime.opset14 as ops
from openvino.frontend import ConversionExtension
#! [py_frontend_extension_ThresholdedReLU_header]

#! [add_extension]
core = ov.Core()

# Use operation type to add operation extension
core.add_extension(Identity)

# or you can add operation extension object which is equivalent form
core.add_extension(ov.OpExtension(Identity))
#! [add_extension]

#! [add_frontend_extension]
# Register more sophisticated mapping with decomposition
def conversion(node):
    input_node = node.get_input(0)
    return Identity(input_node).outputs()

core.add_extension(ConversionExtension("Identity", conversion))
#! [add_frontend_extension]

from utils import get_path_to_extension_library

path_to_extension_lib = get_path_to_extension_library()

#! [add_extension_lib]
core = ov.Core()
# Load extensions library to ov.Core
core.add_extension(path_to_extension_lib)
#! [add_extension_lib]

#! [py_frontend_extension_MyRelu]
from openvino.frontend import OpExtension
core.add_extension(OpExtension("Relu", "MyRelu"))
#! [py_frontend_extension_MyRelu]

#! [py_frontend_extension_ThresholdedReLU]
def conversion(node):
    input_node = node.get_input(0)
    input_type = input_node.get_element_type()
    greater = ops.greater(input_node, ops.constant([node.get_attribute("alpha")], input_type))
    casted = ops.convert(greater, input_type.get_type_name())
    return ops.multiply(input_node, casted).outputs()

core.add_extension(ConversionExtension("ThresholdedRelu", conversion))
#! [py_frontend_extension_ThresholdedReLU]


#! [py_frontend_extension_aten_hardtanh]
import torch
from openvino.frontend import ConversionExtension, NodeContext
from openvino import convert_model


class HardTanh(torch.nn.Module):
    def __init__(self, min_val, max_val):
        super(HardTanh, self).__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, inp):
        return torch.nn.functional.hardtanh(inp, self.min_val, self.max_val)


def convert_hardtanh(node: NodeContext):
    inp = node.get_input(0)
    min_value = node.get_values_from_const_input(1)
    max_value = node.get_values_from_const_input(2)
    return ops.clamp(inp, min_value, max_value).outputs()


model = HardTanh(min_val=0.1, max_val=2.0)
hardtanh_ext = ConversionExtension("aten::hardtanh", convert_hardtanh)
ov_model = convert_model(input_model=model, extension=[hardtanh_ext])
#! [py_frontend_extension_aten_hardtanh]
