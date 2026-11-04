# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Factory functions for ops added to openvino opset17."""
from functools import partial
from typing import Optional

from openvino import Node
from openvino.utils.decorators import nameable_op, unary_op
from openvino.utils.node_factory import _get_node_factory
from openvino.utils.types import NodeInput, as_nodes

_get_node_factory_opset17 = partial(_get_node_factory, "opset17")

# -------------------------------------------- ops ------------------------------------------------


@unary_op
def erfinv(node: NodeInput, name: Optional[str] = None) -> Node:
    """Return node which calculates the inverse error function element-wise on the input tensor.

    :param node: The node providing data for the operation. Must be of floating-point type.
    :param name: The optional name for the new output node.
    :return: The new node performing the element-wise ErfInv operation.
    """
    return _get_node_factory_opset17().create("ErfInv", [node])


@nameable_op
def grouped_matmul(
    mat_a: NodeInput,
    mat_b: NodeInput,
    offsets: Optional[NodeInput] = None,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs Grouped Matrix Multiplication for Mixture of Experts (MoE).

    Computes multiple matrix multiplications where each group processes a subset of the input
    data. Two input combinations are supported:

    - Case 1 (2D x 3D), MoE forward pass, requires ``offsets``:
        - mat_a: (total_tokens, K) - rows partitioned by offsets
        - mat_b: (G, N, K) - per-group weights
        - output: (total_tokens, N)
    - Case 2 (3D x 3D), batched uniform, no offsets:
        - mat_a: (G, M, K) - per-group inputs
        - mat_b: (G, N, K) - per-group weights
        - output: (G, M, N)

    :param mat_a: The first input tensor.
    :param mat_b: The second input tensor with per-group weights.
    :param offsets: 1D tensor of cumulative end-offsets of shape (G,) indicating
                    group boundaries. Required for the 2D x 3D case.
    :param name: The optional name for the new output node.

    :return: The new node performing GroupedMatMul operation.
    """
    inputs = [mat_a, mat_b]
    if offsets is not None:
        inputs.append(offsets)
    return _get_node_factory_opset17().create(
        "GroupedMatMul",
        as_nodes(*inputs, name=name),
    )


@nameable_op
def rgb_to_nv12(
    arg: NodeInput,
    single_plane: bool = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs RGBtoNV12 operation.

    :param arg: The node providing data for the operation in RGB format (NHWC layout).
    :param single_plane: If True, output is single plane NV12 [N, H*3/2, W, 1].
                         If False, output has two planes: Y [N, H, W, 1] and UV [N, H/2, W/2, 2].
    :param name: The optional name for the created output node.
    :return: The new node performing RGBtoNV12 operation.
    """
    return _get_node_factory_opset17().create(
        "RGBtoNV12",
        as_nodes(arg, name=name),
        {"single_plane": single_plane},
    )


@nameable_op
def bgr_to_nv12(
    arg: NodeInput,
    single_plane: bool = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which performs BGRtoNV12 operation.

    :param arg: The node providing data for the operation in BGR format (NHWC layout).
    :param single_plane: If True, output is single plane NV12 [N, H*3/2, W, 1].
                         If False, output has two planes: Y [N, H, W, 1] and UV [N, H/2, W/2, 2].
    :param name: The optional name for the created output node.
    :return: The new node performing BGRtoNV12 operation.
    """
    return _get_node_factory_opset17().create(
        "BGRtoNV12",
        as_nodes(arg, name=name),
        {"single_plane": single_plane},
    )


@nameable_op
def qsa_prepare(
    raw_index_keys: NodeInput,
    past_block_index_keys: NodeInput,
    key_length: NodeInput,
    rope_cos: NodeInput,
    rope_sin: NodeInput,
    k_norm_weight: NodeInput,
    past_incomplete_raw_keys: NodeInput,
    past_incomplete_rope: NodeInput,
    key_block_size: int,
    epsilon: float = 1e-6,
    name: Optional[str] = None,
) -> Node:
    """Return a node which builds one index key per complete key block for sparse attention.

    A block key is the mean of the raw index keys of the ``key_block_size`` consecutive key tokens
    of the block, RMS-normalized with ``k_norm_weight`` and rotated with the RoPE of the first token
    of the block. Only the blocks completed by the new tokens are computed. All inputs are
    positional; a feature is disabled by connecting an empty constant.

    :param raw_index_keys: Raw index key of every new token, shape [N, L, Di].
    :param past_block_index_keys: Block index keys computed by the previous calls, shape [N, Bp, Di].
    :param key_length: Number of valid key tokens of every sequence, shape [N] or [1].
    :param rope_cos: RoPE cos of every new token, shape [N, L, Dr]; empty constant disables the rotation.
    :param rope_sin: RoPE sin of every new token, shape [N, L, Dr]; empty constant disables the rotation.
    :param k_norm_weight: Scale of the block key normalization, shape [Di]; empty constant disables it.
    :param past_incomplete_raw_keys: Raw keys of the past tokens of the incomplete block,
                                     shape [N, key_block_size - 1, Di].
    :param past_incomplete_rope: RoPE cos and sin of the first token of the incomplete block,
                                 shape [N, 2 * Dr]; empty constant if and only if rope_cos and rope_sin are.
    :param key_block_size: Number of consecutive key tokens forming one key block, greater than 1.
    :param epsilon: Value added to the mean square in the RMS normalization.
    :param name: The optional name for the new output node.

    :return: The new node performing QSAPrepare operation.
    """
    inputs = [
        raw_index_keys,
        past_block_index_keys,
        key_length,
        rope_cos,
        rope_sin,
        k_norm_weight,
        past_incomplete_raw_keys,
        past_incomplete_rope,
    ]
    return _get_node_factory_opset17().create(
        "QSAPrepare",
        as_nodes(*inputs, name=name),
        {"key_block_size": key_block_size, "epsilon": epsilon},
    )


@nameable_op
def sparse_attention_indexer(
    index_query: NodeInput,
    block_index_keys: NodeInput,
    key_length: NodeInput,
    index_weights: Optional[NodeInput] = None,
    *,
    key_block_size: int,
    token_budget: int,
    num_selection_heads: int = 1,
    name: Optional[str] = None,
) -> Node:
    """Return a node which selects the key blocks every query token attends to.

    For every query token the operation scores the complete key blocks visible to it and returns the
    indices of the ``token_budget / key_block_size`` best-scoring blocks together with their number.

    :param index_query: Index query heads of shape [N, Hi, L, Di].
    :param block_index_keys: One index key per complete key block, shape [N, B, Di].
    :param key_length: Number of valid key tokens of every sequence, shape [N] or [1].
    :param index_weights: Weight of every index query head for every query token, shape [N, L, Hi].
    :param key_block_size: Number of consecutive key tokens forming one key block.
    :param token_budget: Maximum number of tokens selected from complete blocks for one query.
    :param num_selection_heads: Number of selections per query token.
    :param name: The optional name for the new output node.

    :return: The new node performing SparseAttentionIndexer operation.
    """
    inputs = [index_query, block_index_keys, key_length]
    if index_weights is not None:
        inputs.append(index_weights)
    return _get_node_factory_opset17().create(
        "SparseAttentionIndexer",
        as_nodes(*inputs, name=name),
        {
            "key_block_size": key_block_size,
            "token_budget": token_budget,
            "num_selection_heads": num_selection_heads,
        },
    )


@nameable_op
def sparse_attention(
    query: NodeInput,
    key: NodeInput,
    value: NodeInput,
    block_indices: NodeInput,
    block_count: NodeInput,
    key_length: Optional[NodeInput] = None,
    scale: Optional[NodeInput] = None,
    sinks: Optional[NodeInput] = None,
    *,
    key_block_size: int,
    query_block_size: int = 1,
    causal: bool = True,
    name: Optional[str] = None,
) -> Node:
    """Return a node which computes attention over the key blocks selected for each query.

    Every query token attends only to the tokens of its selected key blocks and, in the causal mode,
    to the tokens of the incomplete block that contains it.

    :param query: Attention query of shape [N, H, L, E].
    :param key: Attention key of shape [N, Hkv, S, E].
    :param value: Attention value of shape [N, Hkv, S, Ev].
    :param block_indices: Selected key blocks of shape [N, Hs, Lq, KB].
    :param block_count: Number of valid entries of every row of block_indices, shape [N, Hs, Lq].
    :param key_length: Number of valid key tokens of every sequence, shape [N] or [1].
    :param scale: Attention scale factor used instead of the default 1 / sqrt(E).
    :param sinks: Attention sink logit of every query head, shape [H]. Requires ``scale``.
    :param key_block_size: Number of consecutive key tokens forming one key block.
    :param query_block_size: Number of consecutive query tokens sharing one selection.
    :param causal: Whether the query tokens are the last key tokens and attend to their
                   incomplete block.
    :param name: The optional name for the new output node.

    :return: The new node performing SparseAttention operation.
    """
    inputs = [query, key, value, block_indices, block_count]
    optional_inputs = [key_length, scale, sinks]
    for position, optional_input in enumerate(optional_inputs):
        if optional_input is None:
            if any(following is not None for following in optional_inputs[position + 1:]):
                raise ValueError("SparseAttention inputs are positional: key_length, scale and sinks "
                                 "can only be omitted from the end.")
            break
        inputs.append(optional_input)
    return _get_node_factory_opset17().create(
        "SparseAttention",
        as_nodes(*inputs, name=name),
        {
            "key_block_size": key_block_size,
            "query_block_size": query_block_size,
            "causal": causal,
        },
    )
