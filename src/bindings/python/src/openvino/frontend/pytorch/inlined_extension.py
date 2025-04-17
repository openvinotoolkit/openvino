# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from typing import Any, Tuple, Type, Union
import torch
import numpy as np
import openvino as ov
from openvino.frontend.pytorch.ts_decoder import InlineConversionExtension
from openvino.frontend.pytorch.utils import pt_to_ov_type_map


class ConstWrap:
    def __init__(self, value: Any):
        self.value = value

    def __eq__(self, x: Any) -> bool:
        return self.value == x

    def __repr__(self) -> str:
        return f'ConstWrap({str(self.value)})'


def unpack(
    packed: Any,
    types: Union[Type, Tuple[Type, ...]],
    index: int = 0
) -> Tuple[Tuple[Any, ...], Any, int]:
    unpacked_result = ()
    if isinstance(packed, tuple):
        packer_result = ()
        for el in packed:
            unpacked, packer, index = unpack(el, types, index)
            unpacked_result += unpacked
            packer_result += (packer,)
    elif isinstance(packed, list):
        packer_result = []
        for el in packed:
            unpacked, packer, index = unpack(el, types, index)
            unpacked_result += unpacked
            packer_result.append(packer)
    elif isinstance(packed, dict):
        packer_result = {}
        for k, v in packed.items():
            unpacked, packer, index = unpack(v, types, index)
            unpacked_result += unpacked
            packer_result[k] = packer
    elif isinstance(packed, types):
        unpacked_result = (packed,)
        packer_result = index
        index += 1
    else:
        packer_result = ConstWrap(packed)
    return unpacked_result, packer_result, index


def pack(unpacked, packer):
    if isinstance(packer, (tuple, list)):
        return type(packer)(pack(unpacked, el) for el in packer)
    if isinstance(packer, dict):
        return {k: pack(unpacked, v) for k, v in packer.items()}
    if isinstance(packer, ConstWrap):
        return packer.value
    return unpacked[packer]


global_counter_id = 0


# makes a custom op class from a func and input/output signatures
def make_custom_op_class(func, input_signature, output_signature, input_packer, output_packer):
    global global_counter_id

    class InlinedCustomOp(ov.Op):
        class_type_info = ov.runtime.DiscreteTypeInfo(
            "InlinedCustomOp", "extension")

        def __init__(self, *args):
            # TODO: What about attributes?
            super().__init__(self, args)
            # `id` attribute distinguishes different instances of the same
            # class, we need it because different instances may have different
            # behavior
            self.attrs = {"id": global_counter_id}
            self.constructor_validate_and_infer_types()

        def evaluate(self, outputs, inputs):
            # TODO: Check memory sharing
            inputs_torch = tuple(torch.from_numpy(input.data)
                                 for input in inputs)
            args, kwargs = pack(inputs_torch, input_packer)
            result = func(*args, **kwargs)
            result, result_packer, _ = unpack(result, torch.Tensor)
            assert result_packer == output_packer
            for i, tensor in enumerate(result):
                if isinstance(tensor, torch.Tensor):
                    ov_t = ov.Tensor(tensor.numpy(force=True), shared_memory=True)
                else:
                    ov_t = ov.Tensor(np.array(tensor), shared_memory=True)
                # TODO: set the output tensor directly without copying
                ov_t.copy_to(outputs[i])
            return True

        def has_evaluate(self, *args):
            return True

        def visit_attributes(self, visitor):
            visitor.on_attributes(self.attrs)
            return True

        def validate_and_infer_types(self):
            # TODO: Validate input signature
            assert output_signature != (), (
                "Operation does not produce any output. "
                "It will be removed from the graph."
            )
            for i, output in enumerate(output_signature):
                self.set_output_type(i, output[0], output[1])

    global_counter_id += 1
    return InlinedCustomOp


def make_signature(args):
    """Convert each torch.Tensor object in args to a tuple (element_type, partial_shape) in OpenVINO terms."""
    # TODO: Extend beyond just tensors
    return tuple((pt_to_ov_type_map[str(arg.dtype)], ov.PartialShape.dynamic(len(arg.shape))) for arg in args)


# Returns a tuple of tuples (element_type, partial_shape) for each argument,
# flattening nested structures if needed, setting all dimensions dynamic
# preserving rank. Currently assumes that all input arguments are torch.Tensor
def make_input_signature(args):
    return make_signature(args)


def make_output_signature(args):
    if args is None:
        args = ()
    if not isinstance(args, tuple):
        args = (args,)
    return make_signature(args)


def make_trampoline_class(func, op, op_attrs):
    class Trampoline(torch.autograd.Function):
        # this is a marker for this type of extension
        target_extension = InlineConversionExtension()

        # This function defines how the operation behaves when called as a
        # part of PyTorch model code in eager execution or while jit.trace
        @staticmethod
        def forward(ctx, *call_args):
            if not op:
                input_signature = make_input_signature(call_args)
            # TODO: Try to trace `func` with the hope to obtain traceable
            # shapes to build more precise `validate_and_infer_types`
            # automatically (unlikely possible)
            packed_args, packed_kwargs = pack(
                call_args, __class__.input_packer)
            result = func(*packed_args, **packed_kwargs)
            result, __class__.output_packer, _ = unpack(result, torch.Tensor)
            if not op:
                output_signature = make_output_signature(result)
                __class__.op = make_custom_op_class(func, input_signature,
                                                    output_signature,
                                                    __class__.input_packer,
                                                    __class__.output_packer)
            else:
                __class__.op = op
            return result

        # Unpack each element that is a tuple, a list, or a dict to a tuple of
        # their values and concatenate together. Build `packer` function to
        # pack the result back to the original nested data types, save this
        # `packer` to class member to use it later in `forward`.
        @staticmethod
        def unpack_inputs(args, kwargs):
            unpacked, __class__.input_packer, _ = unpack(
                (args, kwargs), torch.Tensor)
            return unpacked

        @staticmethod
        def pack_outputs(result):
            return pack(result, __class__.output_packer)

        @staticmethod
        def convert(node_context):
            """Defines how the operation is represented in OpenVINO model graph"""
            inputs = [node_context.get_input(i) for i in range(
                node_context.get_input_size())]
            node = __class__.op(*inputs, **op_attrs)
            # to trigger prim::TupleUnpack bypass in FrontEnd transformation
            node.get_rt_info()['__torch_tuple_unpackable__'] = True
            return node.outputs()

    return Trampoline


def inlined_extension(*args, **op_attrs):
    def make_trampoline(func, op=None):
        def trampoline(*args, **kwargs):
            # Keep trampoline class creation at the point when the function is
            # called to make each time a new trampoline. It is required
            # because `func` is fused inside Trampoline class and can have
            # different behavior from call to call in PyTorch world even if
            # the same op is specified to wrap multiple different functions.
            # It happens due to capturing a global context and values kept in
            # non-traceable part of inputs.
            trampoline = make_trampoline_class(func, op, op_attrs)
            # flattening inputs and unflattening outputs should be visible for
            # PyTorch jit tracer, so it should be called outside of
            # `trampoline.apply` to have clean Tensor-only signature for the
            # custom op. Non-traceable part of inputs (and outputs) is captured
            # as `trampoline` class fields.
            args = trampoline.unpack_inputs(args, kwargs)
            result = trampoline.apply(*args)
            # pack to the expected nested data types here
            return trampoline.pack_outputs(result)
        return trampoline

    if (len(args) == 1 and callable(args[0])
            and not (isinstance(args[0], type)
                     and issubclass(args[0], ov.Op))):
        func = args[0]
        return make_trampoline(func)

    op = args[0]
    return lambda func: make_trampoline(func, op)
