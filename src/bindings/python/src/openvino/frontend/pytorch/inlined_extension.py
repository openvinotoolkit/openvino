# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import inspect
from openvino.frontend.pytorch.ts_decoder import InlineConversionExtension
from openvino.frontend.pytorch.utils import pt_to_ov_type_map
import openvino as ov


class ConstWrap:
    def __init__(self, value):
        self.value = value
    def __eq__(self, x):
        return self.value == x


def unpack(packed, types, index=0):
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
    if isinstance(packer, tuple):
        packed_result = ()
        for el in packer:
            packed = pack(unpacked, el)
            packed_result += (packed,)
    elif isinstance(packer, list):
        packed_result = []
        for el in packer:
            packed = pack(unpacked, el)
            packed_result.append(packed)
    elif isinstance(packer, dict):
        packed_result = {}
        for k, v in packer.items():
            packed = pack(unpacked, v)
            packed_result[k] = packed
    elif isinstance(packer, ConstWrap):
        packed_result = packer.value
    else:
        packed_result = unpacked[packer]
    return packed_result


global_counter_id = 0

# makes a custom op class from a func and input/output signatures
def make_custom_op_class(func, input_signature, output_signature, input_packer, output_packer):
    import torch, numpy
    global global_counter_id
    # print('make_custom_op_class, id =', global_counter_id)
    class InlinedCustomOp(ov.Op):
        class_type_info = ov.runtime.DiscreteTypeInfo("InlinedCustomOp", "extension")

        def __init__(self, *args):
            # TODO: What about attributes?
            super().__init__(self, args)
            self.attrs = {"id": global_counter_id}  # `id` attribute distinguishes different instances of the same class, we need it because different instances may have different behaviour
            #print('output_signature from ctro:', output_signature)
            if output_signature == ():
                # The operation doesn't have outputs, so we need to take extra care to avoid eliminating the op from the graph
                #print('===================== MARKING AS SINK ========================')
                self.get_rt_info()['__sink__'] = True
            #print(f'Made custom op class with id = {self.attrs["id"]}')
            #print(f"Input signature: {input_signature}")
            #print(f"Output signature: {output_signature}")
            self.constructor_validate_and_infer_types()

        def evaluate(self, outputs, inputs):
            # print("called evaluate")
            inputs_torch = tuple(torch.from_numpy(input.data) for input in inputs)   # TODO: Check memory sharing
            args, kwargs = pack(inputs_torch, input_packer)
            result = func(*args, **kwargs)
            result, result_packer, _ = unpack(result, torch.Tensor)
            assert result_packer == output_packer
            for i, tensor in enumerate(result):
                ov.Tensor(numpy.array(tensor), shared_memory=True).copy_to(outputs[i])    # TODO: set the output tensor directly without copying
            return True

        def has_evaluate(self, *args):
            return True

        def visit_attributes(self, visitor):
            visitor.on_attributes(self.attrs)
            return True

        def validate_and_infer_types(self):
            #TODO: Validate input signature
            if output_signature == ():
                # Even when the original wrapped function doesn't give any return value, we need to set some output type to avoid eliminating the op from the graph
                # Data type and shape doesn't matter, so we set default empty tensor of type u8
                self.set_output_type(0, ov.Type.u8, ov.PartialShape([0]))
            else:
                for i, output in enumerate(output_signature):
                    self.set_output_type(i, output[0], output[1])
    global_counter_id += 1
    return InlinedCustomOp


def make_signature(args):
    # TODO: Extend beyond just tensors
    # convert each torch.Tensor object in args to a tuple (element_type, partial_shape) in OpenVINO terms
    return tuple((pt_to_ov_type_map[str(arg.dtype)], ov.PartialShape.dynamic(len(arg.shape))) for arg in args)


# Returns a tuple of tuples (element_type, partial_shape) for each argument, flattening nested structures if needed, setting all dimensions dynamic preserving rank
# Currently assumes that all input arguments are torch.Tensor objects
def make_input_signature(args):
    # TODO: Avoid the current limitation: kwargs parameters should be passed in the same order as the function signature without gaps
    # flatten kwargs relying on the order of the keys
    return make_signature(args)


def make_output_signature(args):
    if args is None:
        # TODO: This case is not really supported by PT FE -- because we don't support ops that do not have outputs, they will be lost
        #print('=================== None PROCESSING ======================')
        args = ()
    if not isinstance(args, tuple):
        args = (args,)
    return make_signature(args)


def make_trampoline_class(func, op, op_attrs):
    import torch
    class Trampoline(torch.autograd.Function):
        target_extension = InlineConversionExtension()  # this is a marker for this type of extension

        # This function defines how the operation behaves when called as a part of PyTorch model code in eager execution or while jit.trace
        @staticmethod
        def forward(ctx, *call_args):  #TODO: what is `ctx`?
            # print('Called through the trampoline')
            if not op:
                input_signature = make_input_signature(call_args)
            # TODO: Try to trace `func` with the hope to obtain tracable shapes to build more precise `validate_and_infer_types` automatically (unlikely possible)
            print('about to call func_target with call_args:', call_args)
            packed_args, packed_kwargs = pack(call_args, __class__.input_packer)
            assert isinstance(packed_args, tuple)
            assert isinstance(packed_kwargs, dict)
            result = func(*packed_args, **packed_kwargs)
            result, __class__.output_packer, _ = unpack(result, torch.Tensor)
            print('output_packer:', __class__.output_packer)
            if not op:
                output_signature = make_output_signature(result)
                #print('about to make custom op class with output signature', output_signature)
                __class__.op = make_custom_op_class(func, input_signature, output_signature, __class__.input_packer, __class__.output_packer)
            else:
                __class__.op = op
            return result

        # Unpack each element that is a tuple, a list, or a dict to a tuple of their values and concatenate together
        # Build `packer` function to pack the result back to the original nested data types, save this `packer` to
        # class member to use it later in `forward`.
        @staticmethod
        def unpack_inputs(args, kwargs):
            unpacked, __class__.input_packer, _ = unpack((args, kwargs), torch.Tensor)
            print('input_packer:', __class__.input_packer)
            return unpacked

        @staticmethod
        def pack_outputs(result):
            return pack(result, __class__.output_packer)

        # This function defines how the operation is represented in OpenVINO model graph
        @staticmethod
        def convert(node_context):
            inputs = [node_context.get_input(i) for i in range(node_context.get_input_size())]
            return __class__.op(*inputs, **op_attrs).outputs()

    return Trampoline


def inlined_extension(*args, **op_attrs):
    def make_trampoline(func, op=None):
        def trampoline(*args, **kwargs):
            # Keep trampoline class creation at the point when the function is called to make each time a new trampoline.
            # It is required because `func` is fused inside Trampoline class and can have different behaviour from call to call in PyTorch world even if
            # the same op is specified to wrap multiple different functions.
            trampoline = make_trampoline_class(func, op, op_attrs)
            print('calling trampoline.apply with args:', args, 'kwargs:', kwargs)
            args = trampoline.unpack_inputs(args, kwargs)
            result = trampoline.apply(*args)
            result = trampoline.pack_outputs(result)
            # pack to the expected nested data types here
            #print('just called trampoline with result:', result)
            return result
        return trampoline

    if len(args) == 1 and callable(args[0]) and not (isinstance(args[0], type) and issubclass(args[0], ov.Op)):
        func = args[0]
        return make_trampoline(func)
    else:
        op = args[0]
        return lambda func: make_trampoline(func, op)
