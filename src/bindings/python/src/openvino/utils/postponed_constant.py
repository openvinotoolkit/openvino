# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, List
import openvino

"""Postponed Constant is a way to materialize a big constant only when it is going to be serialized to IR and then immediately dispose."""
class PostponedConstant(openvino.Op):
    def __init__(self, element_type: openvino.Type, shape: openvino.Shape, maker: Callable[[], openvino.Tensor]) -> None:
        super().__init__(self)
        self.get_rt_info()["postponed_constant"] = True  # value doesn't matter
        self.m_element_type = element_type
        self.m_shape = shape
        self.m_maker = maker
        self.constructor_validate_and_infer_types()

    def evaluate(self, outputs: List[openvino.Tensor], _: List[openvino.Tensor]) -> bool:
        self.m_maker().copy_to(outputs[0])
        #TODO: change to outputs[0] = self.m_maker()
        return True

    def validate_and_infer_types(self) -> None:
        self.set_output_type(0, self.m_element_type, openvino.PartialShape(self.m_shape))

    def clone_with_new_inputs(self, new_inputs):
        return PostponedConstant(self.m_element_type, self.m_shape, self.m_maker)

    def has_evaluate(self):
        return True

# `maker` is a function that returns ov.Tensor that represents a target Constant
def make_postponed_constant(element_type: openvino.Type, shape: openvino.Shape, maker: Callable[[], openvino.Tensor]) -> openvino.Op:
    return PostponedConstant(element_type, shape, maker)
