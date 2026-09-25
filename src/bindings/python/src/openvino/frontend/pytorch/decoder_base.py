# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# mypy: ignore-errors

from openvino import OVAny
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

    def has_converter(self):
        return False
