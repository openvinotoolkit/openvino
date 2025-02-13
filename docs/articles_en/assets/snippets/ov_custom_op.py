# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#

# ! [op:common_include]
from openvino import Op
# ! [op:common_include]



# ! [op:header]
class Identity(Op):
# ! [op:header]

# ! [op:ctor]
    def __init__(self, inputs=None, **attrs):
        super().__init__(self, inputs)
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
        outputs[0].shape = inputs[0].shape
        inputs[0].copy_to(outputs[0])
        return True

    def has_evaluate(self):
        return True
# ! [op:evaluate]

# ! [op:visit_attributes]
    def visit_attributes(self, visitor):
        visitor.on_attributes(self._attrs)
        return True
# ! [op:visit_attributes]
