from typing import Dict

import torch
from torch.nn import Module
from torch._ops import OpOverload

from torch.fx import GraphModule
from torch.fx.node import Node, _get_qualified_name
from torch.fx.passes.operator_support import OperatorSupport
from torch.fx.passes.tools_common import CALLABLE_NODE_OPS

from torch.fx.experimental.proxy_tensor import DecompositionInterpreter
from torch._decomp import decomposition_table
from torch.utils._pytree import tree_flatten, tree_map, tree_unflatten

#from openvino.frontend.pytorch.torchdynamo.openvino_executor import execute

import typing as t

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)

def aten_to_dtype(self, dtype: torch.dtype, **kwargs):
    if len(kwargs) > 0 or not dtype:
        raise RuntimeError("No support for other to.dtype() formats other than to.dtype(self, dtype)")
    return torch._prims.convert_element_type(self, dtype)

# decomposition_table currently contains both aten2aten and aten2prim decomposition
# this is a hack to separate them, as we only need aten2prim decomposition for nvfuser-supported aten graph lowering
aten2aten_decomp = {}
aten2prim_decomp = {}

for op, decomp_fn in decomposition_table.items():
    if "torch._refs" in decomp_fn.__module__:
        aten2prim_decomp[op] = decomp_fn
    else:
        aten2aten_decomp[op] = decomp_fn

aten2aten_decomp_skips = {
    "aten.native_layer_norm_backward.default",
    "aten.embedding_dense_backward.default",   # This is hurting nvfuser's perf
    "aten.addmm.default"
}

for op, decomp_fn in decomposition_table.items():
    if "torch._refs" in decomp_fn.__module__:
        aten2prim_decomp[op] = decomp_fn
    else:
        if str(op) not in aten2aten_decomp_skips:
            aten2aten_decomp[op] = decomp_fn


aten2prim_decomp[torch.ops.aten.to.dtype] = aten_to_dtype


class OperatorSupport(OperatorSupport):
    """
    Operator support for OpenVINO backend.
    Currently, partitioning is based on FX ATen graph. The fused subgraph will latter be decomposed into prims.
    To determine if an ATen ops is supported by nvFuser, we shall check the prim ops used in its ref decomposition.
    Only if all the prim ops in the ref has a nvfuser_impl, we say this Aten op is suppported by nvFuser.
    Note: When adding a rule, please add it to the corresponding section and follow the
    alphabetical order.
    """

    def __init__(self):

        # TODO: current list copied from torch/csrc/jit/codegen/cuda/parser.cpp is incorrect,
        # as that file is solely for TorchScript and doesn't represent the actual status
        # whether operation would be runnable by primTorch+nvFuser.
        # We will iterate on this list to reflect the the reality.
        """
        support_dict = {
            # ===============================================================
            # call_function aten
            # ===============================================================
            # Following supported aten ops is copied from torch/csrc/jit/codegen/cuda/parser.cpp
            # TODO: might need to update according to supported input types

            "torch.ops.aten.relu": None,
            "torch.ops.aten.relu_": None,
            "torch.ops.aten.conv2d": None,
            "torch.ops.aten._convolution": None,
            "torch.ops.aten.convolution": None,
            "torch.ops.aten.batch_norm": None,
            "torch.ops.aten.layer_norm": None,
            "torch.ops.aten.add": None,
            "torch.ops.aten.add_": None,
            "torch.ops.aten.mul": None,
            "torch.ops.aten.mul_": None,
            "torch.ops.aten.div": None,
            "torch.ops.aten.floordiv": None,
            "torch.ops.aten.tanh": None,
            "torch.ops.aten.elu": None,
            "torch.ops.aten.sigmoid": None,
            "torch.ops.aten.gelu": None,
            "torch.ops.aten.sqrt": None,
            "torch.ops.aten.abs": None,
            "torch.ops.aten.square": None,
            "torch.ops.aten.hardtanh": None,
            "torch.ops.aten.hardtanh_": None,
            "torch.ops.aten.hardsigmoid": None,
            "torch.ops.aten.hardswish": None,
            "torch.ops.aten.hardswish_": None,
            "torch.ops.aten.silu_": None,
            "torch.ops.aten.relu6": None,
            "torch.ops.aten.softmax": None,
            "torch.ops.aten.matmul": None,
            "torch.ops.aten.mm": None,
            "torch.ops.aten.linear": None,
            "torch.ops.aten.max_pool2d": None,
            "torch.ops.aten.avg_pool2d": None,
            "torch.ops.aten.adaptive_avg_pool2d": None,
            "torch.ops.aten.adaptive_max_pool2d": None,
            #"torch.ops.aten.max_pool2d_with_indices": None,
            "torch.ops.aten.mean": None,
            "torch.ops.aten.flatten": None,
            #"torch.ops.prim.NumToTensor": None,
            "torch.ops.aten.contiguous": None,
            "torch.ops.aten.as_tensor": None,
            "torch.ops.aten.Int": None,
            "torch.ops.aten.to": None,
            "torch.ops.aten.permute": None,
            "torch.ops.aten.embedding": None,
            "torch.ops.aten.transpose": None,
            "torch.ops.aten.size": None,
            "torch.ops.aten.view": None,
            "torch.ops.aten.unsqueeze": None,
            "torch.ops.aten.rsub": None,
            "torch.ops.aten.slice": None,
            #"torch.ops.prim.Loop": None,
            #"torch.ops.prim.If": None,
            #"torch.ops.prim.Constant": None,
            "torch.ops.aten.dim": None,
            "torch.ops.aten.reciprocal": None,
            "torch.ops.aten.sub": None,
            "torch.ops.aten.eq": None,
            "torch.ops.aten.ne": None,
            "torch.ops.aten.gt": None,
            "torch.ops.aten.lt": None,
            "torch.ops.aten.neg": None,
            #"torch.ops.prim.TupleConstruct": None,
            "torch.ops.aten.append": None,
            "getattr": None,
            "_operator.getitem": None,
        }
        """
        # Just added Resnet50 supported iterations
        support_dict = {
            "_operator.getitem": None,
            "torch.ops.aten._adaptive_avg_pool2d.default": None,
            "torch.ops.aten._softmax.default": None,
            "torch.ops.aten._to_copy.default": None,
            "torch.ops.aten._unsafe_view.default": None,
            "torch.ops.aten._unsafe_view.default": None,
            "torch.ops.aten.add.Tensor": None,
            "torch.ops.aten.add_.Tensor": None,
            "torch.ops.aten.addmm.default": None,
            "torch.ops.aten.arange.start": None,
            "torch.ops.aten.avg_pool2d.default": None,
            "torch.ops.aten.bitwise_and.Tensor": None,
            "torch.ops.aten.bmm.default": None,
            "torch.ops.aten.cat.default": None,
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
            "torch.ops.aten.eq.Tensor": None,
            "torch.ops.aten.exp.default": None,
            "torch.ops.aten.expand.default": None,
            "torch.ops.aten.full.default": None,
            #"torch.ops.aten.gather.default": None,
            "torch.ops.aten.gelu.default": None,
            #"torch.ops.aten.gt.Scalar": None,
            "torch.ops.aten.hardsigmoid.default": None,
            "torch.ops.aten.hardswish_.default": None,
            "torch.ops.aten.hardtanh_.default": None,
            "torch.ops.aten.lift_fresh_copy.default": None,
            "torch.ops.aten.log.default": None,
            #"torch.ops.aten.logsumexp.default": None,
            #"torch.ops.aten.max.dim": None,
            "torch.ops.aten.max_pool2d_with_indices.default": None,
            "torch.ops.aten.mean.dim": None,
            "torch.ops.aten.mm.default": None,
            "torch.ops.aten.mul.Tensor": None,
            "torch.ops.aten.native_batch_norm.default": None,
            "torch.ops.aten.native_group_norm.default": None,
            "torch.ops.aten.native_layer_norm.default": None,
            "torch.ops.aten.neg.default": None,
            #"torch.ops.aten.new_ones.default" : None,
            "torch.ops.aten.permute.default": None,
            "torch.ops.aten.pow.Tensor_Scalar": None,
            "torch.ops.aten.relu.default": None,
            "torch.ops.aten.relu_.default": None,
            "torch.ops.aten.rsub.Scalar": None,
            "torch.ops.aten.select.int": None,
            "torch.ops.aten.sigmoid.default": None,
            "torch.ops.aten.silu.default": None,
            "torch.ops.aten.silu_.default": None,
            "torch.ops.aten.sin.default": None,
            "torch.ops.aten.slice.Tensor": None,
            "torch.ops.aten.split.Tensor": None,
            #"torch.ops.aten.stack.default": None,
            "torch.ops.aten.sub.default": None,
            "torch.ops.aten.sub.Tensor": None,
            "torch.ops.aten.t.default": None,
            "torch.ops.aten.tanh.default": None,
            "torch.ops.aten.transpose.int": None,
            "torch.ops.aten.unsqueeze.default": None,
            "torch.ops.aten.upsample_nearest2d.default": None,
            "torch.ops.aten.view.default": None,
            "torch.ops.aten.where.self": None,
            "torch.ops.aten.zeros_like.default": None,
        }

        super().__init__(support_dict)

    def is_node_supported(
        self, submodules: t.Mapping[str, Module], node: Node
    ) -> bool:

        # OpenVINO FX subgraph should be purely functional
        if node.op not in CALLABLE_NODE_OPS:
            return False

        print("target: ", node.target)
        # ops in supported_dict doesn't have overload name
        # use overloadpacket's qualified_name for OpOverload
        if isinstance(node.target, OpOverload):
            target = _get_qualified_name(node.target.overloadpacket)

            if target in self._support_dict:
                return True

        return super().is_node_supported(submodules, node)
