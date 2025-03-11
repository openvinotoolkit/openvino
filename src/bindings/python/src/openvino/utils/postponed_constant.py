# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Callable, List, Optional
from openvino import Op, Type, Shape, Tensor, PartialShape


class PostponedConstant(Op):
    """Postponed Constant is a way to materialize a big constant only when it is going to be serialized to IR and then immediately dispose."""
    def __init__(self, element_type: Type, shape: Shape, maker: Callable[[Tensor], None], name: Optional[str] = None) -> None:
        super().__init__(self)
        self.get_rt_info()["postponed_constant"] = True  # value doesn't matter
        self.m_element_type = element_type
        self.m_shape = shape
        self.m_maker = maker
        if name is not None:
            self.friendly_name = name
        self.constructor_validate_and_infer_types()

    def evaluate(self, outputs: List[Tensor], _: List[Tensor]) -> bool:
        self.m_maker(outputs[0])
        return True

    def validate_and_infer_types(self) -> None:
        self.set_output_type(0, self.m_element_type, PartialShape(self.m_shape))

    def clone_with_new_inputs(self, new_inputs: List[Tensor]) -> Op:
        return PostponedConstant(self.m_element_type, self.m_shape, self.m_maker, self.friendly_name)

    def has_evaluate(self) -> bool:
        return True


# `maker` is a function that returns ov.Tensor that represents a target Constant
def make_postponed_constant(element_type: Type, shape: Shape, maker: Callable[[Tensor], None], name: Optional[str] = None) -> Op:
    return PostponedConstant(element_type, shape, maker, name)
