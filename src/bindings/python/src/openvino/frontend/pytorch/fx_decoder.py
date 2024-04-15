# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
from openvino.runtime import op, PartialShape, Type as OVType, OVAny, Shape
from openvino.frontend.pytorch.utils import make_constant, fetch_attr, pt_to_ov_type_map, torch_tensor_to_ov_const

import torch

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class TorchFXPythonDecoder (Decoder):

    def __init__(self, pt_module, fx_gm=None, nodes=None, mark_node_callback=None, input_shapes=[], input_types=[]):
        Decoder.__init__(self)
        self.mark_node_callback = mark_node_callback
        # We store every decoder created by this decoder so that all them are not deleted until the first decoder is deleted
        self.m_decoders = []
        self.pt_module = pt_module
        self.fx_gm = fx_gm if fx_gm is not None else pt_module
        self.input_types = [OVAny(pt_to_ov_type_map[str(t)])
                            for t in input_types]
        self.input_shapes = input_shapes

        self._input_signature = []

        if issubclass(type(pt_module), torch.fx.graph_module.GraphModule):

            self._input_is_list = None
            self._nodes = list(pt_module.graph.nodes)
            self._inputs = []
            self._outputs = []
            found_types = []
            found_shapes = []
            for i in range(len(self._nodes)):
                if self._nodes[i].op == 'placeholder':
                    self._inputs.append(i)
                    value = self._nodes[i]
                    self._input_signature.append(value.name)
                    if hasattr(value, "meta") and ('tensor_meta' in value.meta.keys()) and value.meta['tensor_meta']:
                        found_shapes.append(value.meta['tensor_meta'].shape)
                        found_types.append(
                            OVAny(pt_to_ov_type_map[str(value.meta['tensor_meta'].dtype)]))
                    else:
                        found_shapes.append(None)
                        found_types.append(None)
                elif self._nodes[i].op == 'output':
                    # Instead of putting output index, refer to its target
                    uargs = self.unpack_containers(self._nodes[i].args)
                    self._outputs = [(arg[0], self._nodes.index(arg[1]))
                                     for arg in uargs if arg[1] is not None]

            if not input_shapes or len(input_shapes) == 0:
                self.input_shapes = found_shapes
            if not input_types or len(input_types) == 0:
                self.input_types = found_types

        elif issubclass(type(pt_module), torch.fx.Node):

            self._nodes = nodes  # passed from outer context

            # FIXME: Quadratic complexity nodes*nodes considering the outer loop over all nodes
            self._outputs = [("", self._nodes.index(pt_module))]

            # None in inputs mean the input is inlined or None (also considered inlined)
            self._inputs = [self._nodes.index(
                arg) if arg in self._nodes else (arg,) for arg in pt_module.args]

            # FIXME: Find a better way to pass nested tuples to OV frontend. This is a temporary solution to flatten arguments.
            new_inputs = []
            self.input_types = []
            for i in range(len(pt_module.args)):
                if isinstance(pt_module.args[i], (list, tuple)) and any([isinstance(a, torch.fx.Node) for a in pt_module.args[i]]):
                    for arg in pt_module.args[i]:
                        if arg in self._nodes:
                            new_inputs.append(self._nodes.index(arg))
                        else:
                            new_inputs.append((arg,))
                        self.input_types.append(OVAny(DecoderType.List(
                            TorchFXPythonDecoder.get_type_for_value(arg))))
                else:
                    v = self._inputs[i]
                    new_inputs.append(v)
                    self.input_types.append(
                        TorchFXPythonDecoder.get_type_for_value(v[0] if isinstance(v, tuple) else self._nodes[v]))
            self._inputs = new_inputs

    def inputs(self):
        # Consider 0 a special case which may mean the input is inlined, but not guaranteed
        return [x if not isinstance(x, tuple) else 0 for x in self._inputs]

    def is_input_inlined(self, index):
        return isinstance(self._inputs[index], tuple)

    @staticmethod
    def unpack_containers(arg):
        if isinstance(arg, (tuple, list)):
            res = []
            for e in arg:
                res.extend(TorchFXPythonDecoder.unpack_containers(e))
            return res
        elif isinstance(arg, dict):
            res = []
            for k, e in arg.items():
                unpacked = TorchFXPythonDecoder.unpack_containers(e)
                if len(unpacked) == 1:
                    unpacked[0] = (k, unpacked[0][1])
                res.extend(unpacked)
            return res
        else:
            return [("", arg)]

    @staticmethod
    def arg_to_constant(arg):
        if isinstance(arg, list):
            if len(arg) > 0:
                return make_constant(pt_to_ov_type_map[type(
                    arg[0]).__name__], Shape([len(arg)]), arg)
            else:
                # TODO: which type should we use if list is empty? Need a signaling value here
                return make_constant(OVType.i32, Shape([0]), [])
        elif isinstance(arg, bool):
            return make_constant(OVType.boolean, Shape([]), [arg])
        elif isinstance(arg, int):
            return make_constant(OVType.i64, Shape([]), [arg])
        elif isinstance(arg, float):
            return make_constant(OVType.f32, Shape([]), [arg])
        return None

    def inlined_input(self, index):
        assert index < len(self._inputs), "Requested input doesn't exist"
        assert isinstance(
            self._inputs[index], tuple), "Requested input which is not inlined"
        assert self._inputs[index][0] is not None, "Requested None inlined input"
        constant = None
        arg = self._inputs[index][0]
        constant = self.arg_to_constant(arg)

        assert constant is not None, f"Constant wasn't created for inlined input {index}"
        return constant.outputs()

    def input(self, index):  # TODO: remove
        return self.inputs()[index]  # TODO: find specialized method

    def get_input_debug_name(self, index):
        return "input"+str(index)

    def get_input_signature_name(self, index: int) -> str:
        if self._input_signature is not None and index < len(self._input_signature):
            return self._input_signature[index]
        return self.get_input_debug_name(index)

    def get_input_shape(self, index):
        if index < len(self.input_shapes) and self.input_shapes[index] is not None:
            return PartialShape(self.input_shapes[index])
        input = self._raw_input(index)
        return self.get_shape_for_value(input)

    def get_input_strides(self, index: int) -> list:
        raw_input = self._raw_input(index)
        if isinstance(raw_input, torch.fx.node.Node) and hasattr(raw_input, "meta"):
            meta = raw_input.meta
            if "tensor_meta" in meta and hasattr(meta["tensor_meta"], "stride"):
                strides = list(meta["tensor_meta"].stride)
                if strides:
                    return strides
        return []

    def get_input_type(self, index):
        if index < len(self.input_types) and self.input_types[index] is not None:
            return self.input_types[index]
        input = self._raw_input(index)
        return self.get_type_for_value(input)

    def get_output_debug_name(self, index):
        if self._outputs is not None and index < len(self._outputs) and self._outputs[index][0]:
            return self._outputs[index][0]
        name = getattr(self.pt_module, "name", "output")
        return name + ":" + str(index)

    def get_output_shape(self, index):
        output = self._raw_output(index)
        return self.get_shape_for_value(output)

    def get_output_type(self, index):
        output = self._raw_output(index)
        return self.get_type_for_value(output)

    def get_shape_for_value(self, value):
        if value and hasattr(value, "meta") and ('tensor_meta' in value.meta.keys()):
            if value.meta['tensor_meta']:
                return PartialShape(len(value.meta['tensor_meta'].shape) * [-1])
        return PartialShape.dynamic()

    @staticmethod
    def get_type_for_value(value):
        if issubclass(type(value), torch.fx.Node):
            if ('tensor_meta' in value.meta.keys()):
                if value.meta['tensor_meta'] and isinstance(value.meta['tensor_meta'], torch.Tensor):
                    pt_type = value.meta['tensor_meta'].dtype
                    if str(pt_type) in pt_to_ov_type_map:
                        ov_type = pt_to_ov_type_map[str(pt_type)]
                        return OVAny(ov_type)
            return OVAny(OVType.dynamic)
        elif isinstance(value, int):
            return OVAny(DecoderType.PyScalar(OVAny(OVType.i64)))
        elif isinstance(value, float):
            return OVAny(DecoderType.PyScalar(OVAny(OVType.f32)))
        elif isinstance(value, bool):
            return OVAny(DecoderType.PyScalar(OVAny(OVType.boolean)))
        return OVAny(OVType.dynamic)

    def get_attribute(self, name):
        if name in self.pt_module.kwargs:
            attr = self.pt_module.kwargs[name]
            if isinstance(attr, torch.dtype):
                return OVAny(pt_to_ov_type_map[str(attr)])
            if isinstance(attr, torch.device):
                return OVAny(attr.type)
            if isinstance(attr, str):
                return OVAny(attr)
            # Numeric attrs convert to Constant
            constant = self.arg_to_constant(attr)
            if constant is not None:
                return OVAny(constant.output(0))
            # so that has_attribute return True if attribute exist
            return OVAny(DecoderType.PyNone())
        return OVAny(None)

    def get_named_input(self, name):
        """
        Returns id of kwargs input. Such input can be Node or a constant value,
        this function is only used for to return node index. If the input is
        constant, get_attribute should be used.
        """
        if name in self.pt_module.kwargs:
            arg = self.pt_module.kwargs[name]
            if isinstance(arg, torch.fx.Node):
                return self._nodes.index(arg)
        raise RuntimeError("This input is not a Node")

    def get_subgraph_size(self):
        if issubclass(type(self.pt_module), torch.fx.Node):
            return 0
        return len(self.get_subgraphs()) if hasattr(self.pt_module, 'blocks') else 1

    def decoder_type_name(self) -> str:
        return "fx"

    def visit_subgraph(self, node_visitor):
        # make sure topological order is satisfied
        for node in self._nodes:
            if node.op == 'placeholder' or node.op == 'output':
                continue  # skipping non-operational nodes
            if node.op == 'call_function' and str(node.target) in ["aten._assert_async.msg"]:
                continue
            decoder = TorchFXPythonDecoder(
                node, self.fx_gm, self._nodes, mark_node_callback=self.mark_node_callback)
            self.m_decoders.append(decoder)
            node_visitor(decoder)

    def get_subgraphs(self):
        if issubclass(type(self.pt_module), torch.fx.Node):
            return []
        return list(self.pt_module.blocks())

    def get_subgraph_decoder(self, index):
        decoder = TorchFXPythonDecoder(self.get_subgraphs()[index],
                                       self.fx_gm,
                                       mark_node_callback=self.mark_node_callback)
        self.m_decoders.append(decoder)
        return decoder

    def get_op_type(self):
        if self.pt_module.op == 'call_function':
            return str(self.pt_module.target)
        elif self.pt_module.op == 'get_attr':
            return 'get_attr'  # FIXME should be aligned with get_attr from TS implementation
        else:
            return 'UNKNOWN_TYPE_' + str(self.pt_module.op)

    def get_schema(self):
        return 'NONE'

    def outputs(self):
        return [o[1] for o in self._outputs]

    def _raw_outputs(self):
        return [self._nodes[x[1]] for x in self._outputs]

    def _raw_output(self, index):
        return self._raw_outputs()[index]

    def _raw_inputs(self):
        return [self._nodes[x] if not isinstance(x, tuple) and x < len(self._nodes) else x[0] for x in self._inputs]

    def _raw_input(self, index):
        return self._raw_inputs()[index]

    def num_of_outputs(self):
        return len(self.outputs())

    def output(self, index):
        return self.outputs()[index]

    def mark_node(self, node):
        name = self.get_op_type()
        if "FrameworkNode" not in node.get_type_name():
            name += "/" + node.get_type_name()
        node.set_friendly_name(self.pt_module.name + "/" + name)
        if self.mark_node_callback is not None:
            self.mark_node_callback(self, node)
        return node

    def as_constant(self):
        assert self.pt_module.op == 'get_attr', "Only get_attr is supported"
        # Extract Constant from FX module field
        ret = fetch_attr(self.fx_gm, self.pt_module.target)
        ov_const = torch_tensor_to_ov_const(ret, shared_memory=True)
        return ov_const.outputs()

    def as_string(self):
        return None

    def input_is_none(self, index):
        if index >= len(self._inputs) or (isinstance(self._inputs[index], tuple) and self._inputs[index][0] is None):
            return True
        else:
            r_input = self._raw_input(index)
            return str(type(r_input)) in ['torch.NoneType', 'NoneType']

    def debug(self):
        self.pt_module.print()

    def may_produce_alias(self, in_index: int, out_index: int) -> bool:
        return False
