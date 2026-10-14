# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from openvino import OVAny, Type as OVType
from openvino.frontend.pytorch.py_pytorch_frontend import _FrontEndPytorchDecoder as Decoder


class TorchDecoderBase(Decoder):
    """Logic shared by the TorchScript and FX decoders.

    Holds the parts of the decoder interface that do not depend on which PyTorch IR is
    being decoded: friendly-name construction, the input-signature fallback, and the
    defaults for capabilities that only one of the two IRs has. Everything that reads
    ``torch`` data structures belongs in the subclasses.
    """

    def __init__(self) -> None:
        super().__init__()
        # Every decoder produced by this decoder is kept here so that none of them is
        # destroyed while the first one is still alive.
        self.m_decoders = []
        self._input_signature = None
        self._default_float_type = OVType.f32

    # -- friendly names ----------------------------------------------------------------

    def _node_name_prefix(self):
        """Scope-like prefix for friendly names, or None when the node has no scope."""
        return None

    def mark_node(self, node):
        """Associate an OpenVINO node with the framework node being decoded.

        Subclasses may override this to attach framework data to ``node``; they should
        call ``super().mark_node(node)`` so that the friendly name is still applied.
        """
        name = self.get_op_type()
        if "FrameworkNode" not in node.get_type_name():
            name += "/" + node.get_type_name()
        prefix = self._node_name_prefix()
        node.set_friendly_name(f"{prefix}/{name}" if prefix else name)
        return node

    # -- inputs and outputs ------------------------------------------------------------

    def get_input_signature_name(self, index: int) -> str:
        signature = self._input_signature
        if signature is not None and index < len(signature):
            return signature[index]
        return self.get_input_debug_name(index)

    def output(self, index: int):
        return self.outputs()[index]

    def num_of_outputs(self):
        return len(self.outputs())

    def is_input_inlined(self, index):
        return False

    def get_inlined_input_decoder(self, index):
        return None

    # -- named inputs: FX kwargs only ---------------------------------------------------

    def get_attribute(self, name):
        return OVAny(None)

    def get_named_input(self, name):
        raise RuntimeError(f"{type(self).__name__} has no named inputs")

    # -- optional capabilities ----------------------------------------------------------

    def get_subgraphs(self) -> list:
        return []

    def get_subgraph_size(self) -> int:
        return len(self.get_subgraphs())

    def as_string(self):
        return None

    def may_produce_alias(self, in_index: int, out_index: int) -> bool:
        return False

    def get_rt_info(self):
        return {}

    # -- torch's default float type -----------------------------------------------------

    def get_default_float_type(self):
        """Torch's default float type for the model, as an OpenVINO type.

        The dtype ``torch.set_default_dtype`` had when the model was created, which is the type
        of its float weights and activations. PyTorch's own default, f32, unless the decoder was
        created with another value. When it isn't f32, the frontend converts the model to f32
        activations with the weights kept in that type. Read from the top-level decoder only.
        """
        return self._default_float_type

    def _set_default_float_type(self, value) -> None:
        """Set the value returned by get_default_float_type(); ``value`` is an OpenVINO type or None (f32)."""
        if value is None:
            value = OVType.f32
        if not isinstance(value, OVType) or not value.is_real():
            raise ValueError(f"default_float_type must be a floating-point type, got {value!r}")
        self._default_float_type = value

    def has_converter(self):
        return False
