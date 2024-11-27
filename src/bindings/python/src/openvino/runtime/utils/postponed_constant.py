# -*- coding: utf-8 -*-
# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import openvino

"""Postponed Constant is a way to materialize a big constant only when it is going to be serialized to IR and then immediately dispose."""


# `maker` is a function that returns ov.Tensor that represents a target Constant
def make_postponed_constant(element_type, shape, maker):
    class PostponedConstant(openvino.Op):
        class_type_info = openvino.runtime.DiscreteTypeInfo("PostponedConstant", "extension")

        def __init__(self):
            super().__init__(self)
            self.get_rt_info()["postponed_constant"] = True  # value doesn't matter
            self.m_element_type = element_type
            self.m_shape = shape
            self.constructor_validate_and_infer_types()

        def get_type_info(self):
            return PostponedConstant.class_type_info

        def evaluate(self, outputs, _):
            maker().copy_to(outputs[0])
            return True

        def clone_with_new_inputs(self, _):
            return PostponedConstant()

        def validate_and_infer_types(self):
            self.set_output_type(0, self.m_element_type, openvino.PartialShape(self.m_shape))

    return PostponedConstant()
