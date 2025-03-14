# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

import inspect
from openvino.frontend.pytorch.ts_decoder import InlineConversionExtension
from openvino.frontend.pytorch.utils import pt_to_ov_type_map
import openvino as ov

global_counter_id = 0

# makes a custom op class from a func and input/output signatures
def make_custom_op_class(func, input_signature, output_signature):
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
            inputs_torch = (torch.from_numpy(input.data) for input in inputs)   # TODO: Check memory sharing
            result = func(*inputs_torch)
            if result is None:
                result = ()
            if not isinstance(result, tuple):
                result = (result,)
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
def make_input_signature(args, kwargs):
    # TODO: Avoid the current limitation: kwargs parameters should be passed in the same order as the function signature without gaps
    # flatten kwargs relying on the order of the keys
    assert not kwargs, "Keyword arguments are not supported yet"
    return make_signature(args + tuple(kwargs.values()))


def make_output_signature(args):
    if args is None:
        # TODO: This case is not really supported by PT FE -- because we don't support ops that do not have outputs, they will be lost
        #print('=================== None PROCESSING ======================')
        args = ()
    if not isinstance(args, tuple):
        args = (args,)
    return make_signature(args)


def is_class_method(obj):
    if not inspect.isfunction(obj) and not inspect.ismethod(obj):
        return False
    argspec = inspect.getfullargspec(obj)
    if argspec.args and argspec.args[0] == 'self':
        return True
    else:
        return False


def make_trampoline_class(func, op, op_attrs):
    import torch
    class Trampoline(torch.autograd.Function):
        target_extension = InlineConversionExtension()  # this is a marker for this type of extension

        # This function defines how the operation behaves when called as a part of PyTorch model code in eager execution or while jit.trace
        @staticmethod
        def forward(ctx, *call_args, **call_kwargs):  #TODO: what is `ctx`?
            # print('Called through the trampoline')
            func_target = func
            if not op:
                if is_class_method(func):
                    self_obj = call_args[0]
                    call_args = call_args[1:]
                    wrapped = lambda *distil_args, **distil_kwargs: func(self_obj, *distil_args, **distil_kwargs)
                    func_target = wrapped
                input_signature = make_input_signature(call_args, call_kwargs)
            # TODO: Try to trace `func` with the hope to obtain tracable shapes to build more precise `validate_and_infer_types` automatically (unlikely possible)
            result = func_target(*call_args, **call_kwargs)
            if not op:
                output_signature = make_output_signature(result)
                #print('about to make custom op class with output signature', output_signature)
                __class__.op = make_custom_op_class(func_target, input_signature, output_signature)
            else:
                __class__.op = op
            return result

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
            result = trampoline.apply(*args, **kwargs)
            #print('just called trampoline with result:', result)
            return result
        return trampoline

    if len(args) == 1 and callable(args[0]) and not (isinstance(args[0], type) and issubclass(args[0], ov.Op)):
        func = args[0]
        return make_trampoline(func)
    else:
        op = args[0]
        return lambda func: make_trampoline(func, op)
