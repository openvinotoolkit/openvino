# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Op

class CustomOp(Op):
    # class_type_info = ov.runtime.DiscreteTypeInfo("PagedAttentionExtension", "extension")
    def __init__(self, inputs):
        super().__init__()
        self.set_arguments(inputs)
        self.validate()
    def validate_and_infer_types(self):
        self.set_output_type(0, self.get_input_element_type(0), self.get_input_partial_shape(0))
    def clone_with_new_inputs(self, new_inputs):
        node = CustomOp(new_inputs)
        return node
    def get_type_info(self):
        return CustomOp.class_type_info
    def evaluate(self, outputs, inputs):
        #FIXME: Stub
        print(inputs[0].data.shape)
        outputs[0] = inputs[0]
        return True
    def has_evaluate(self):
        return True


def test_custom_op():
    custom_op = CustomOp(inputs=5)