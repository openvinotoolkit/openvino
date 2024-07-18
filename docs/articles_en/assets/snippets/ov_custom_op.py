# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# ! [op:common_include]
import numpy as np
from openvino import Op
from openvino.runtime import DiscreteTypeInfo
# ! [op:common_include]



# ! [op:header]
class Identity(Op):
    class_type_info = DiscreteTypeInfo("Identity", "extension")

    def get_type_info(self):
        return Identity.class_type_info
# ! [op:header]

# ! [op:ctor]
    def __init__(self, inputs, attrs=None):
        super().__init__(self)
        self.set_arguments(inputs)
        self.constructor_validate_and_infer_types()
        if attrs is not None:
            self._attrs = attrs
# ! [op:ctor]

# ! [op:validate]
    def validate_and_infer_types(self):
        self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))
# ! [op:validate]

# ! [op:copy]
    def clone_with_new_inputs(self, new_inputs):
        return Identity(new_inputs)
# ! [op:copy]

# ! [op:evaluate]
    def evaluate(self, outputs, inputs):
        if np.array_equal(outputs[0].data, inputs[0].data):  # Nothing to do
            return True
        outputs[0].shape = inputs[0].shape
        inputs[0].copy_to(outputs[0])
        return True

    def has_evaluate(self):
        return True
# ! [op:evaluate]

# ! [op:visit_attributes]
    def visit_attributes(self, visitor):
        if hasattr(self, "_attrs"):
            visitor.on_attributes(self._attrs)
        return True
# ! [op:visit_attributes]
