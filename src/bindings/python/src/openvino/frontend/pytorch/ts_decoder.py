# Copyright (C) 2018-2023 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder
from openvino.frontend.pytorch.py_pytorch_frontend import _Type as DecoderType
from openvino.runtime import op, PartialShape, Type as OVType, OVAny
from openvino.frontend.pytorch.utils import ivalue_to_constant, get_value_from_getattr, pt_to_ov_type_map, prepare_example_inputs_and_model, convert_quantized_tensor
from openvino.runtime import opset11 as ops
from openvino.frontend.pytorch import gptq

import typing
import torch


class TorchScriptPythonDecoder (Decoder):
    def __init__(self, pt_module, graph_element=None, example_input=None, alias_db=None, shared_memory=True, skip_freeze=False):
        Decoder.__init__(self)
        # We store every decoder created by this decoder so that all them are not deleted until the first decoder is deleted
        self.m_decoders = []
        self._input_signature = None
        self._shared_memory = shared_memory
        self._input_is_list = False
        if graph_element is None:
            try:
                pt_module = self._get_scripted_model(pt_module, example_input, skip_freeze)
            except Exception as e:
                if example_input is not None:
                    msg = "tracing"
                    help_msg = "Please check correctness of provided 'example_input'. "
                    "Sometimes models can be converted in scripted mode, please try running "
                    "conversion without 'example_input'."
                else:
                    msg = "scripting"
                    help_msg = "\nTracing sometimes provide better results, please provide valid 'example_input' argument."
                raise RuntimeError(
                    f"Couldn't get TorchScript module by {msg}. With exception:\n{e}\n{help_msg} "
                    "You can also provide TorchScript module that you obtained"
                    " yourself, please refer to PyTorch documentation: "
                    "https://pytorch.org/tutorials/beginner/Intro_to_TorchScript_tutorial.html.")
            self.graph_element = pt_module.inlined_graph
            self.alias_db = self.graph_element.alias_db()
        else:
            self.graph_element = graph_element
            self.alias_db = alias_db
        self.pt_module = pt_module
        self.raw_inputs = list(self.graph_element.inputs())
        self.raw_outputs = list(self.graph_element.outputs())
        if self._input_signature is not None:
            if "self" in self.raw_inputs[0].debugName():
                self._input_signature.insert(0, "self")
            if 0 < len(self._input_signature) < len(self.raw_inputs):
                # last input is args input, we need to multiply that name by number of extra inputs
                self._input_signature = self._input_signature[:-1]
                n = len(self._input_signature)
                for i in range(len(self.raw_inputs) - n):
                    self._input_signature.append(self.raw_inputs[i + n].debugName())

        if isinstance(self.graph_element, torch.Graph):
            self._transform_tensor_list_constants_to_listconstruct(self.graph_element)
            self._transform_optional_constants(self.graph_element)

    @staticmethod
    def _get_preserved_attributes(model) -> list:
        preserved_attributes = []
        for name, module in model.named_modules():
            if hasattr(module, "weight"):
                if module.weight is not None and module.weight.dtype in [torch.int8, torch.uint8]:
                    preserved_attributes.append(name)
        return preserved_attributes

    def _get_scripted_model(self, pt_module, example_inputs=None, skip_freeze=False):
        import torch
        import inspect

        if isinstance(pt_module, torch.nn.Module):
            pt_module.eval()
        input_signature = None
        if isinstance(pt_module, torch.nn.Module) and not isinstance(pt_module, (torch.jit._trace.TopLevelTracedModule, torch.jit._script.RecursiveScriptModule)):
            # input params is dictionary contains input names and their signature values (type hints and default values if any)
            input_params = inspect.signature(pt_module.forward if hasattr(pt_module, "forward") else pt_module.__call__).parameters
            input_signature = list(input_params)
            if example_inputs is None:
                scripted = torch.jit.script(pt_module)
            else:
                input_parameters, input_signature, pt_module, self._input_is_list = prepare_example_inputs_and_model(
                    example_inputs, input_params, pt_module)
                gptq_patched = False

                if gptq.detect_gptq_model(pt_module):
                    try:
                        gptq.patch_model(pt_module)
                        gptq_patched = True
                    except Exception as error:
                        print('[ WARNING ] Failed patching of AutoGPTQ model. Error message:\n', error)
                        print('[ WARNING ] Tracing of the model will likely be unsuccesfull or incorrect')
                        gptq.unpatch_model(pt_module)
                        gptq_patched = False

                try:
                    scripted = torch.jit.trace(
                        pt_module, **input_parameters, strict=False)
                finally:
                    if gptq_patched:
                        gptq.unpatch_model(pt_module)

            if not skip_freeze:
                ops_kind_no_freeze = ["quantize", "aten::as_strided"]
                for n in scripted.inlined_graph.nodes():
                    # TODO: switch off freezing for all traced models
                    if any(kind in n.kind() for kind in ops_kind_no_freeze):
                        # do not freeze quantized models
                        skip_freeze = True
                        break
                    elif "aten::to" in n.kind():
                        first_input = next(n.inputs())
                        if first_input.node().kind() == "prim::Constant":
                            ivalue = first_input.toIValue()
                            if isinstance(ivalue, torch.Tensor) and ivalue.dtype in [torch.bfloat16, torch.float16]:
                                # do not freeze models with compressed constants
                                skip_freeze = True
                                break
            if not skip_freeze:
                preserved_attrs = self._get_preserved_attributes(scripted)
                f_model = torch.jit.freeze(scripted, preserved_attrs=preserved_attrs)
            else:
                f_model = scripted
        else:
            f_model = pt_module

        self._input_signature = input_signature
        return f_model

    def inputs(self) -> list:
        return [x.unique() for x in self.raw_inputs]

    def get_input(self, index: int):
        return self.inputs()[index]

    def get_input_debug_name(self, index: int) -> str:
        return self._raw_input(index).debugName()

    def get_input_signature_name(self, index: int) -> str:
        if self._input_signature is not None and index < len(self._input_signature):
            return self._input_signature[index]
        return self.get_input_debug_name(index)

    def get_input_shape(self, index: int):
        raw_input = self._raw_input(index)
        return self.get_shape_for_value(raw_input)

    def get_input_strides(self, index: int) -> typing.List[int]:
        raw_input = self._raw_input(index)
        if isinstance(raw_input, torch.Value):
            inp_type = raw_input.type()
            if isinstance(inp_type, torch.TensorType):
                strides = inp_type.strides()
                if strides:
                    return strides
        return []

    def get_input_type(self, index: int):
        raw_input = self._raw_input(index)
        return self.get_type_for_value(raw_input)

    def get_output_debug_name(self, index: int) -> str:
        return self._raw_output(index).debugName()

    def get_output_shape(self, index: int):
        output = self._raw_output(index)
        return self.get_shape_for_value(output)

    def get_output_type(self, index: int):
        output = self._raw_output(index)
        return self.get_type_for_value(output)

    def _get_known_type_for_value(self, pt_type):
        """Returns known/unknown types wrapped as OVAny."""
        # Check for simple scalar types first
        if pt_type is None:
            return OVAny(OVType.dynamic)
        # TODO: Don't use str, use native types
        if str(pt_type) in pt_to_ov_type_map:
            return OVAny(pt_to_ov_type_map[str(pt_type)])
        elif isinstance(pt_type, torch.TensorType):
            # Tensor type, parse element type
            return OVAny(DecoderType.Tensor(self._get_known_type_for_value(pt_type.dtype())))
        elif isinstance(pt_type, torch.ListType):
            element_type = pt_type.getElementType()
            return OVAny(DecoderType.List(self._get_known_type_for_value(element_type)))
        elif isinstance(pt_type, (torch.StringType, torch.DeviceObjType)):
            return OVAny(DecoderType.Str())
        elif isinstance(pt_type, torch.NoneType):
            return OVAny(DecoderType.PyNone())
        else:
            # Not yet recognized
            return OVAny(OVType.dynamic)

    def get_shape_for_value(self, value: torch.Value):
        if value.isCompleteTensor():
            # We avoid static shapes, they don't generalize on other inputs
            ps = PartialShape([-1] * len(value.type().sizes()))
            return ps
        else:
            # TODO: Recognize types that we can represent as a nested constructs with objects from DecoderType
            # If recognized, return scalar instead of dynamic. Scalar means a single value of that custom type.
            # See get_type_for_value for reference
            pass
        return PartialShape.dynamic()

    def get_type_for_value(self, value: torch.Value):
        full_type = self._get_known_type_for_value(value.type())
        return full_type

    def get_subgraph_size(self) -> int:
        if isinstance(self.graph_element, torch.Node):
            return len(self.get_subgraphs())
        else:
            return 1

    def visit_subgraph(self, node_visitor) -> None:
        # make sure topological order is satisfied
        for node in self.graph_element.nodes():
            decoder = TorchScriptPythonDecoder(self.pt_module, node, alias_db=self.alias_db, shared_memory=self._shared_memory)
            self.m_decoders.append(decoder)
            node_visitor(decoder)

    def decoder_type_name(self) -> str:
        return "ts"

    def get_subgraphs(self) -> list:
        if self.graph_element.kind() == "prim::PythonOp":
            if "Subgraph" in self.graph_element.attributeNames():
                assert isinstance(self.graph_element, torch.Node), "Graph element must be of type torch.Node."
                return [getattr(self.graph_element, self.graph_element.kindOf("Subgraph"))("Subgraph")]
            else:
                # Attribute "Subgraph" is only available if Graph was created using tracing.
                # TODO Find way to extract subgraph for scripted Graph.
                return []
        return list(self.graph_element.blocks())

    def get_subgraph_decoder(self, index: int):
        decoder = TorchScriptPythonDecoder(self.pt_module, self.get_subgraphs()[index], alias_db=self.alias_db, shared_memory=self._shared_memory)
        self.m_decoders.append(decoder)
        return decoder

    def get_op_type(self) -> str:
        assert isinstance(self.graph_element, torch.Node), "Function can be called only when self.graph_element is of type torch.Node"
        return self.graph_element.kind()

    def get_schema(self) -> str:
        return self.graph_element.schema()

    def outputs(self) -> list:
        return [x.unique() for x in self.raw_outputs]

    def _raw_output(self, index: int):
        return self.raw_outputs[index]

    def _raw_input(self, index: int):
        return self.raw_inputs[index]

    def num_of_outputs(self):
        return len(self.raw_outputs)

    def output(self, index: int):
        return self.outputs()[index]

    def mark_node(self, node):
        name = self.graph_element.kind()
        if "FrameworkNode" not in node.get_type_name():
            name += "/" + node.get_type_name()
        if self.graph_element.scopeName():
            node.set_friendly_name(self.graph_element.scopeName().split("/")[-1] + "/" + name)
        else:
            node.set_friendly_name(name)
        return node

    def try_decode_get_attr(self):
        pt_value = get_value_from_getattr(self.graph_element, self.pt_module)
        assert pt_value is not None, "Couldn't retrieve value from prim::GetAttr"
        if isinstance(pt_value, torch.ScriptObject):
            # We assume this is __torch__.torch.classes.quantized.Conv2dPackedParamsBase or __torch__.torch.classes.quantized.LinearPackedParamsBase
            # TODO: but can be anything. Figure a better way to distinguish
            weight, bias = pt_value.unpack()
            res = convert_quantized_tensor(weight, self._shared_memory)
            if isinstance(bias, torch.Tensor):
                res += ivalue_to_constant(bias)
            else:
                res += ops.convert_like(ivalue_to_constant(torch.zeros(1))[0], res[0]).outputs()
            try:
                # these params exist only for conv params
                stride = pt_value.stride()
                padding = pt_value.padding()
                dilation = pt_value.dilation()
                groups = pt_value.groups()
                res += ivalue_to_constant(stride, shared_memory=self._shared_memory)
                res += ivalue_to_constant(padding, shared_memory=self._shared_memory)
                res += ivalue_to_constant(dilation, shared_memory=self._shared_memory)
                res += ivalue_to_constant(groups, shared_memory=self._shared_memory)
            except:
                pass
            return res
        elif not isinstance(pt_value, (torch.jit.ScriptModule, torch.jit.TracedModule)):
            return ivalue_to_constant(pt_value, shared_memory=self._shared_memory)
        else:
            return []

    def as_constant(self):
        if not isinstance(self.graph_element, torch.Node):
            return None
        if not self.get_op_type() == "prim::Constant":
            return None
        pt_value = self._raw_output(0)
        pt_type = pt_value.type()
        if isinstance(pt_type, torch.TensorType):
            return ivalue_to_constant(pt_value.toIValue(), shared_memory=self._shared_memory)
        if isinstance(pt_type, torch.ListType):
            return self._as_constant_list(pt_value)
        return ivalue_to_constant(pt_value.toIValue(), shared_memory=self._shared_memory)

    def as_string(self):
        if self.get_op_type() == "prim::Constant":
            pt_value = self._raw_output(0)
            if str(pt_value.type()) in ["torch.StringType", "str"]:
                return pt_value.toIValue()
            elif str(pt_value.type()) == "Device":
                return pt_value.toIValue().type
        elif self.get_op_type() == "prim::device":
            return self._get_device_string()
        return None

    @staticmethod
    def _as_constant_list(pt_value: torch.Value):
        # For now it is treat a list as a 1D tensor; it is required by converters to avoid need to massively
        # rewrite them in that part where constant attributes are queried
        pt_element_type = str(pt_value.type().getElementType())
        ivalue = pt_value.toIValue()
        is_known_type = pt_element_type in pt_to_ov_type_map

        if is_known_type:
            ovtype = pt_to_ov_type_map[pt_element_type]
            ovshape = PartialShape([len(ivalue)])
            ov_const = op.Constant(ovtype, ovshape.get_shape(), ivalue)
            return ov_const.outputs()

    def _get_device_string(self) -> str:
        assert self.graph_element.kind() == "prim::device", "This function can be called for prim::device node."
        value = self.raw_inputs[0]
        if value.type().isSubtypeOf(torch.TensorType.get()):
            tensor = typing.cast(torch.TensorType, value.type())
            device = tensor.device()
            if device:
                return str(device)
        # Device cannot be statically determined.
        return "cpu"

    def input_is_none(self, index: int) -> bool:
        if index >= len(self.inputs()) or self._raw_input(index) is None:
            return True
        else:
            r_input = self._raw_input(index)
            if str(r_input.type()) in ["torch.NoneType", "NoneType"]:
                return True
            else:
                in_node = r_input.node()
                if in_node.kind() == "prim::GetAttr":
                    pt_value = get_value_from_getattr(in_node, self.pt_module)
                    return pt_value is None
        return False

    def may_produce_alias(self, in_index: int, out_index: int) -> bool:
        if self.get_op_type() in ["aten::conv1d", "aten::conv2d", "aten::conv3d", "aten::matmul"]:
            # AliasDB::may_contain_alias sometimes return True for tensors produced by convnd, we have to workaround that
            return False
        try:
            return self.alias_db.may_contain_alias(self._raw_input(in_index), self._raw_output(out_index))
        except:
            # Sometimes pytorch fails to get result with IndexError exception while these indexes exist in node
            return False

    def inlined_inputs(self, index):
        return []

    @staticmethod
    def _transform_tensor_list_constants_to_listconstruct(graph: torch.Graph):
        # Function replaces prim::Constant containing List of Tensors with
        # prim::ListConstruct containing prim::Constant Tensors.
        assert isinstance(graph, torch.Graph), "Function can be called only with parameters of type torch.Graph."
        for node in graph.nodes():
            if node.kind() != "prim::Constant":
                continue
            output_type = node.output().type()
            allowed_types = [
                output_type.isSubtypeOf(torch.ListType.ofTensors()),
                output_type.isSubtypeOf(torch.ListType(torch.OptionalType.ofTensor())),
            ]
            if not any(allowed_types):
                continue
            const_inputs = []
            for val in node.output().toIValue():
                const_input = graph.insertConstant(val)
                const_input.node().moveBefore(node)
                const_input.node().copyMetadata(node)
                const_inputs.append(const_input)

            replacement = graph.create("prim::ListConstruct", const_inputs)
            replacement.insertBefore(node)
            replacement.output().setType(torch.ListType.ofTensors())
            replacement.copyMetadata(node)
            node.output().replaceAllUsesWith(replacement.output())

    @staticmethod
    def _transform_optional_constants(graph: torch.Graph):
        # Function replaces prim::Constant containing torch.OptionalType with
        # prim::Constant containing torch.NoneType or type of IValue.
        assert isinstance(graph, torch.Graph), "Function can be called only with parameters of type torch.Graph."
        for node in graph.nodes():
            if node.kind() != "prim::Constant":
                continue
            output_type = node.output().type()
            if not isinstance(output_type, torch.OptionalType):
                continue
            value = node.output().toIValue()
            const_input = graph.insertConstant(value)
            const_input.node().moveBefore(node)
            const_input.node().copyMetadata(node)
            node.output().replaceAllUsesWith(const_input)
