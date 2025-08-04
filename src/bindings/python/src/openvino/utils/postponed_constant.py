# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0


from typing import Optional, Union, cast
from collections.abc import Callable
from openvino import Op, Type, Shape, Tensor, PartialShape, TensorVector


class PostponedConstant(Op):
    """Postponed Constant is a way to materialize a big constant only when it is going to be serialized to IR and then immediately dispose."""
    def __init__(self, element_type: Type, shape: Shape, maker: Union[Callable[[], Tensor], Callable[[Tensor], None]], name: Optional[str] = None) -> None:
        """Creates a PostponedConstant.

        :param element_type: Element type of the constant.
        :type element_type: openvino.Type
        :param shape: Shape of the constant.
        :type shape: openvino.Shape
        :param maker: A callable that returns a Tensor or modifies the provided Tensor to represent the constant.
                    Note: It's recommended to use a callable without arguments (returns Tensor) to avoid unnecessary tensor data copies.
        :type maker: Union[Callable[[], Tensor], Callable[[Tensor], None]]
        :param name: Optional name for the constant.
        :type name: Optional[str]

        :Example of a maker that returns a Tensor:

        .. code-block:: python

            class Maker:
                def __call__(self) -> ov.Tensor:
                    tensor_data = np.array([2, 2, 2, 2], dtype=np.float32)
                    return ov.Tensor(tensor_data)
        """
        super().__init__(self)
        self.get_rt_info()["postponed_constant"] = True  # value doesn't matter
        self.m_element_type = element_type
        self.m_shape = shape
        self.m_maker = maker
        if name is not None:
            self.friendly_name = name
        self.constructor_validate_and_infer_types()

    def evaluate(self, outputs: TensorVector, _: list[Tensor]) -> bool:  # type: ignore
        num_args = self.m_maker.__call__.__code__.co_argcount
        if num_args == 1:
            maker = cast(Callable[[], Tensor], self.m_maker)
            outputs[0] = maker()
        else:
            maker = cast(Callable[[Tensor], None], self.m_maker)
            maker(outputs[0])
        return True

    def validate_and_infer_types(self) -> None:
        self.set_output_type(0, self.m_element_type, PartialShape(self.m_shape))

    def clone_with_new_inputs(self, new_inputs: list[Tensor]) -> Op:
        return PostponedConstant(self.m_element_type, self.m_shape, self.m_maker, self.friendly_name)

    def has_evaluate(self) -> bool:
        return True


# `maker` is a function that returns ov.Tensor that represents a target Constant
def make_postponed_constant(element_type: Type, shape: Shape, maker: Union[Callable[[], Tensor], Callable[[Tensor], None]], name: Optional[str] = None) -> Op:
    return PostponedConstant(element_type, shape, maker, name)
