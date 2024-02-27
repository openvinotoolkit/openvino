# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# flake8: noqa
# mypy: ignore-errors

from typing import Dict

import torch
from torch.nn import Module
from torch._ops import OpOverload

from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS
from openvino.frontend.pytorch.torchdynamo.backend_utils import _get_disabled_ops

import typing as t
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

class OperatorSupport(OperatorSupport):
    """
    Operator support for OpenVINO backend.
    """

    def __init__(self, options):
        support_dict = {
            "_operator.getitem": None,
            "torch.ops.aten._adaptive_avg_pool2d.default": None,
            "torch.ops.aten._log_softmax.default": None,
            "torch.ops.aten._softmax.default": None,
            "torch.ops.aten._to_copy.default": None,
            "torch.ops.aten._unsafe_view.default": None,
            "torch.ops.aten._unsafe_view.default": None,
            "torch.ops.aten.add.Scalar": None,
            "torch.ops.aten.add.Tensor": None,
            "torch.ops.aten.add_.Tensor": None,
            "torch.ops.aten.addmm.default": None,
            "torch.ops.aten.amax.default": None,
            "torch.ops.aten.arange.start": None,
            "torch.ops.aten.arange.default": None,
            "torch.ops.aten.argmax.default": None,
            "torch.ops.aten.avg_pool2d.default": None,
            "torch.ops.aten.baddbmm.default": None,
            "torch.ops.aten.bitwise_and.Tensor": None,
            "torch.ops.aten.bmm.default": None,
            "torch.ops.aten.cat.default": None,
            "torch.ops.aten.clamp_min.default": None,
            "torch.ops.aten.clone.default": None,
            "torch.ops.aten.convolution.default": None,
            "torch.ops.aten.copy_.default": None,
            "torch.ops.aten.cos.default": None,
            "torch.ops.aten.cumsum.default": None,
            "torch.ops.aten.detach.default": None,
            "torch.ops.aten.div.Scalar": None,
            "torch.ops.aten.div.Tensor": None,
            "torch.ops.aten.embedding.default": None,
            "torch.ops.aten.empty.memory_format": None,
            "torch.ops.aten.erf.default": None,
            "torch.ops.aten.eq.Scalar": None,
            "torch.ops.aten.eq.Tensor": None,
            "torch.ops.aten.exp.default": None,
            "torch.ops.aten.expand.default": None,
            "torch.ops.aten.fill.Scalar": None,
            "torch.ops.aten.full.default": None,
            "torch.ops.aten.gather.default": None,
            "torch.ops.aten.gelu.default": None,
            "torch.ops.aten.gt.Scalar": None,
            "torch.ops.aten.hardsigmoid.default": None,
            "torch.ops.aten.hardswish_.default": None,
            "torch.ops.aten.hardtanh_.default": None,
            "torch.ops.aten.index.Tensor": None,
            "torch.ops.aten.leaky_relu_.default": None,
            "torch.ops.aten.lift_fresh_copy.default": None,
            "torch.ops.aten.linalg_vector_norm.default": None,
            "torch.ops.aten.lt.Tensor": None,
            "torch.ops.aten.log.default": None,
            "torch.ops.aten.log_sigmoid_forward.default": None,
            "torch.ops.aten.logsumexp.default": None,
            "torch.ops.aten.masked_fill_.Scalar": None,
            "torch.ops.aten.masked_fill.Tensor": None,
            "torch.ops.aten.max.dim": None,
            "torch.ops.aten.max_pool2d_with_indices.default": None,
            "torch.ops.aten.mean.dim": None,
            "torch.ops.aten.mm.default": None,
            "torch.ops.aten.mul.Scalar": None,
            "torch.ops.aten.mul.Tensor": None,
            "torch.ops.aten.native_batch_norm.default": None,
            "torch.ops.aten._native_batch_norm_legit.default": None,
            "torch.ops.aten._native_batch_norm_legit_no_training.default": None,
            "torch.ops.aten.native_group_norm.default": None,
            "torch.ops.aten.native_layer_norm.default": None,
            "torch.ops.aten.new_full.default": None,
            "torch.ops.aten.neg.default": None,
            "torch.ops.aten.new_ones.default": None,
            "torch.ops.aten.permute.default": None,
            "torch.ops.aten.pow.Tensor_Scalar": None,
            "torch.ops.aten.relu.default": None,
            "torch.ops.aten.relu_.default": None,
            "torch.ops.aten.rsqrt.default": None,
            "torch.ops.aten.rsub.Scalar": None,
            "torch.ops.aten._scaled_dot_product_flash_attention.default": None,
            "torch.ops.aten.scalar_tensor.default": None,
            "torch.ops.aten.select.int": None,
            "torch.ops.aten.sigmoid.default": None,
            "torch.ops.aten.silu.default": None,
            "torch.ops.aten.silu_.default": None,
            "torch.ops.aten.sin.default": None,
            "torch.ops.aten.slice.Tensor": None,
            "torch.ops.aten.split.Tensor": None,
            "torch.ops.aten.squeeze.dim": None,
            "torch.ops.aten.squeeze.dims": None,
            "torch.ops.aten.stack.default": None,
            "torch.ops.aten.sub.default": None,
            "torch.ops.aten.sub.Tensor": None,
            "torch.ops.aten.sum.dim_IntList": None,
            "torch.ops.aten.t.default": None,
            "torch.ops.aten.tanh.default": None,
            "torch.ops.aten.transpose.int": None,
            "torch.ops.aten.unbind.int": None,
            "torch.ops.aten.unsqueeze.default": None,
            "torch.ops.aten.upsample_nearest2d.default": None,
            "torch.ops.aten.var_mean.correction": None,
            "torch.ops.aten.view.default": None,
            "torch.ops.aten.where.self": None,
            "torch.ops.aten.zeros_like.default": None,
        }

        for op in _get_disabled_ops(options):
            del support_dict[op]

        super().__init__(support_dict)

    def is_node_supported(self, submodules: t.Mapping[str, Module], node: Node) -> bool:
        # OpenVINO FX subgraph should be purely functional
        if node.op not in CALLABLE_NODE_OPS:
            return False

        # ops in supported_dict doesn't have overload name
        # use overloadpacket's qualified_name for OpOverload
        if isinstance(node.target, OpOverload):
            target = _get_qualified_name(node.target.overloadpacket)

            if target in self._support_dict:
                return True

        return super().is_node_supported(submodules, node)
