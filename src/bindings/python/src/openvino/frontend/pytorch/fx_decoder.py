# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
from openvino.runtime import op, PartialShape, Type as OVType, OVAny, Shape
from openvino.frontend.pytorch.utils import maybe_convert_max_int, make_constant, fetch_attr, pt_to_ov_type_map, torch_tensor_to_ov_const

import torch


class TorchFXPythonDecoder (Decoder):

    def __init__(self, pt_module, fx_gm, nodes=None, mark_node_callback=None, input_shapes=[], input_types=[]):
        Decoder.__init__(self)
        self.mark_node_callback = mark_node_callback
        # We store every decoder created by this decoder so that all them are not deleted until the first decoder is deleted
        self.m_decoders = []
        self.pt_module = pt_module
        self.fx_gm = fx_gm
        self.input_types = [OVAny(pt_to_ov_type_map[str(t)])
                            for t in input_types]
        self.input_shapes = input_shapes

        self._input_signature = []

        if issubclass(type(pt_module), torch.fx.graph_module.GraphModule):

            self._input_is_list = None
            self._nodes = list(pt_module.graph.nodes)
            self._inputs = []
            self._outputs = []
            for i in range(len(self._nodes)):
                if self._nodes[i].op == 'placeholder':
                    self._inputs.append(i)
                    self._input_signature.append(self._nodes[i].name)
                elif self._nodes[i].op == 'output':
                    # Instead of putting output index, refer to its target
                    uargs = self.unpack_containers(self._nodes[i].args)
                    self._outputs = [(arg[0], self._nodes.index(arg[1])) for arg in uargs if arg[1] is not None]

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
                    new_inputs.append(self._inputs[i])
                    self.input_types.append(
                        TorchFXPythonDecoder.get_type_for_value(self._inputs[i]))
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
            arg = maybe_convert_max_int(arg)
            return make_constant(OVType.i32, Shape(
                []), [arg])  # TODO: i32? why not i64?
        elif isinstance(arg, float):
            return make_constant(OVType.f32, Shape(
                []), [arg])  # TODO: f32? why not f64?
        return None

    def inlined_input(self, index):
        assert index < len(self._inputs), "Requested input doesn't exist"
        assert isinstance(
            self._inputs[index], tuple), "Requested input which is not inlined"
        assert self._inputs[index][0] is not None, "Requested None inlined input"
        constant = None
        arg = self._inputs[index][0]
        constant = self.arg_to_constant(arg)

        assert constant is not None, "Constant wasn't created for inlined input"
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
        if index < len(self.input_shapes):
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
        if index < len(self.input_types):
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
            else:
                return OVAny(OVType.dynamic)
        elif isinstance(value, int):
            return OVAny(OVType.i32)
        elif isinstance(value, float):
            return OVAny(OVType.f32)
        elif isinstance(value, bool):
            return OVAny(OVType.boolean)
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
        decoder = TorchFXPythonDecoder(self.get_subgraphs(
        )[index], self.fx_gm, mark_node_callback=self.mark_node_callback)
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
        return ''
        return self.pt_module.schema()

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
        node.set_friendly_name(name)
        if self.mark_node_callback is not None:
            self.mark_node_callback(self, node)
        return node

    def as_constant(self):

        if self.pt_module.op == 'get_attr':
            # Extract Constant from FX module field
            ret = fetch_attr(self.fx_gm, self.pt_module.target)
            ov_const = torch_tensor_to_ov_const(ret, shared_memory=True)
            return ov_const.outputs()

        if not self.get_op_type() == 'prim::Constant':
            return None
        pt_value = self._raw_output(0)

        pt_type_class = pt_value.type().__class__
        if pt_type_class is torch.TensorType:
            return self.as_constant_tensor(pt_value)
        if pt_type_class is torch.ListType:
            return self.as_constant_list(pt_value)
        if str(pt_value.type()) in ['torch.int32', 'int']:
            return make_constant(OVType.i32, Shape([]), [pt_value.toIValue()]).outputs()
        if str(pt_value.type()) in ['torch.float', 'torch.FloatType', 'float']:
            return make_constant(OVType.f32, Shape([]), [pt_value.toIValue()]).outputs()
        if str(pt_value.type()) in ['torch.bool', 'bool']:
            return make_constant(OVType.boolean, Shape([]), [pt_value.toIValue()]).outputs()

        return None

    def as_string(self):
        if not self.get_op_type() == 'prim::Constant':
            return None
        pt_value = self._raw_output(0)

        if str(pt_value.type()) in ['torch.StringType', 'str']:
            return pt_value.toIValue()
        return None

    def as_constant_tensor(self, pt_value):
        ivalue = pt_value.toIValue()
        if pt_value.isCompleteTensor():
            try:
                ivalue = ivalue.to(
                    memory_format=torch.contiguous_format).detach().cpu()
            except:
                print("[ WARNING ] Tensor couldn't detach")
            if str(pt_value.type().dtype()) in pt_to_py_type_map:
                # Constant interpretation doesn't respect new-full type of PT
                # It recognizes only tensors, and give lists as 1D tensors, and scalars as Tensor scalars
                # So only tensor-type constants are supported
                ovshape = PartialShape(pt_value.type().sizes())
                ovtype = pt_to_ov_type_map[str(pt_value.type().dtype())]

                # TODO: try-except here is a temporary WA for issues with data_ptr that we currently cannot predict; provide better solution
                try:
                    # this is only possible with adding a new ctor for Constant Python binding
                    # TODO Check strides and pass them somehow
                    values = ivalue.data_ptr()
                    ov_const = make_constant(
                        ovtype, ovshape.get_shape(), values)
                except:
                    # old variant that makes a slow data copying
                    print(
                        f"[ WARNING ] Constant wasn't able to convert from data_ptr.")
                    values = ivalue.flatten().tolist()
                    ov_const = make_constant(
                        ovtype, ovshape.get_shape(), values)
                return ov_const.outputs()
        else:
            # Incomplete tensor can be scalar
            if isinstance(ivalue, float):
                return make_constant(OVType.f32, Shape([]), [ivalue]).outputs()
            if isinstance(ivalue, int):
                return make_constant(OVType.i32, Shape([]), [ivalue]).outputs()
            if isinstance(ivalue, bool):
                return make_constant(OVType.boolean, Shape([]), [ivalue]).outputs()

            # TODO: verify that it correctly reads incomplete consts
            if str(ivalue.type()) in pt_to_ov_type_map:
                try:
                    ovshape = PartialShape(ivalue.size())
                    ovtype = pt_to_ov_type_map[str(ivalue.type())]
                    ov_const = make_constant(
                        ovtype, ovshape.get_shape(), ivalue.data_ptr())
                except:
                    # old variant that makes a slow data copying
                    print(
                        f"[ WARNING ] Constant wasn't able to convert from data_ptr.")
                    nvalues = ivalue.numpy(force=True)
                    ovtype = np_to_ov_type_map[str(nvalues.dtype)]
                    ovshape = PartialShape(nvalues.shape)
                    ov_const = make_constant(
                        ovtype, ovshape.get_shape(), nvalues.flatten().tolist())
                return ov_const.outputs()
        return None

    def as_constant_list(self, pt_value):
        # For now it is treat a list as a 1D tensor; it is required by converters to avoid need to massively rewrite them in that part where constant attributes are queried
        pt_element_type = str(pt_value.type().getElementType())
        ivalue = pt_value.toIValue()
        is_known_type = pt_element_type in pt_to_ov_type_map

        # WA to broken ov.Type
        # Detect integer list and process it with a dedicated method
        # TODO: Fix ov.Type and remove this WA
        # if pt_to_py_type_map[pt_element_type] == 'int':
        #    self.as_constant_list_of_ints(ovshape = PartialShape([len(ivalue)]), ivalue)
        # End of WA to broken ov.Type

        if is_known_type:
            ovtype = pt_to_ov_type_map[pt_element_type]
            ovshape = PartialShape([len(ivalue)])
            ov_const = make_constant(ovtype, ovshape.get_shape(), ivalue)
            return ov_const.outputs()

    def input_is_none(self, index):
        if index >= len(self._inputs) or (isinstance(self._inputs[index], tuple) and self._inputs[index][0] is None):
            return True
        else:
            r_input = self._raw_input(index)
            return str(type(r_input)) in ['torch.NoneType', 'NoneType']

    def debug(self):
        self.pt_module.print()

    def inlined_inputs(self, index):
        result = []
        for i in range(len(self._inputs)):
            if isinstance(self._inputs[i], tuple):
                constant = None
                arg = self._inputs[i][0]
                if isinstance(arg, list):
                    if len(arg) > 0:
                        constant = make_constant(pt_to_ov_type_map[type(
                            arg[0]).__name__], Shape([len(arg)]), arg)
                    else:
                        # TODO: which type should we use if list is empty? Need a signaling value here
                        constant = make_constant(int, Shape([0]), [])
                elif isinstance(arg, bool):
                    constant = make_constant(OVType.boolean, Shape([]), [arg])
                elif isinstance(arg, int):
                    arg = maybe_convert_max_int(arg)
                    constant = make_constant(OVType.i32, Shape(
                        []), [arg])  # TODO: i32? why not i64?
                elif isinstance(arg, float):
                    constant = make_constant(OVType.f32, Shape(
                        []), [arg])  # TODO: f32? why not f64?

                if constant is None:
                    if arg is None:
                        self._inputs[i] = None
                else:
                    assert len(constant.outputs()) == 1
                    result.append(constant.outputs()[0])
                    self._inputs[i] = index
                    index += 1
        return result

    def may_produce_alias(self, in_index: int, out_index: int) -> bool:
        if self.get_op_type() in ["aten::conv1d", "aten::conv2d", "aten::conv3d", "aten::matmul"]:
            # AliasDB::may_contain_alias sometimes return True for tensors produced by convnd, we have to workaround that
            return False
        try:
            return self.alias_db.may_contain_alias(self._raw_input(in_index), self._raw_output(out_index))
        except:
            # Sometimes pytorch fails to get result with IndexError exception while these indexes exist in node
            return False
