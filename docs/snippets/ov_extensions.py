# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

import openvino.runtime as ov

#! [py_frontend_extension_ThresholdedReLU_header]
import openvino.runtime.opset8 as ops
from openvino.frontend import ConversionExtension
#! [py_frontend_extension_ThresholdedReLU_header]

#! [add_extension]
# Not implemented
#! [add_extension]

#! [add_frontend_extension]
# Not implemented
#! [add_frontend_extension]

#! [add_extension_lib]
core = ov.Core()
# Load extensions library to ov.Core
core.add_extension("libopenvino_template_extension.so")
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
