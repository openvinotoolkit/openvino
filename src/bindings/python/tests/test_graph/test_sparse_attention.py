# -*- coding: utf-8 -*-
# Copyright (C) 2018-2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from openvino import Type, PartialShape
from openvino.opset15 import constant, parameter
import numpy as np
import openvino.opset17 as ov
import pytest


def test_sparse_attention():
    query = parameter([2, 24, 5, 64], name="query", dtype=Type.f32)
    key = parameter([2, 2, 12, 64], name="key", dtype=Type.f32)
    value = parameter([2, 2, 12, 32], name="value", dtype=Type.f32)
    block_indices = parameter([2, 1, 5, 4], name="block_indices", dtype=Type.i32)
    block_count = parameter([2, 1, 5], name="block_count", dtype=Type.i32)
    node = ov.sparse_attention(query, key, value, block_indices, block_count, key_block_size=4)

    assert node.get_type_name() == "SparseAttention"
    assert node.get_output_size() == 1
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_output_partial_shape(0) == PartialShape([2, 24, 5, 32])


@pytest.mark.parametrize("dtype_indices", [Type.i32, Type.i64])
def test_sparse_attention_with_key_length_scale_and_sinks(dtype_indices):
    query = parameter([2, 24, 1, 64], name="query", dtype=Type.f32)
    key = parameter([2, 2, 12, 64], name="key", dtype=Type.f32)
    value = parameter([2, 2, 12, 32], name="value", dtype=Type.f32)
    block_indices = parameter([2, 1, 1, 4], name="block_indices", dtype=dtype_indices)
    block_count = parameter([2, 1, 1], name="block_count", dtype=dtype_indices)
    key_length = parameter([2], name="key_length", dtype=dtype_indices)
    scale = parameter([], name="scale", dtype=Type.f32)
    sinks = parameter([24], name="sinks", dtype=Type.f32)
    node = ov.sparse_attention(query, key, value, block_indices, block_count,
                               key_length, scale, sinks, key_block_size=4)

    assert node.get_type_name() == "SparseAttention"
    assert len(node.inputs()) == 8
    assert node.get_output_partial_shape(0) == PartialShape([2, 24, 1, 32])


def test_sparse_attention_non_causal():
    query = parameter([1, 12, 256, 128], name="query", dtype=Type.f32)
    key = parameter([1, 12, 512, 128], name="key", dtype=Type.f32)
    value = parameter([1, 12, 512, 128], name="value", dtype=Type.f32)
    block_indices = parameter([1, 12, 2, 4], name="block_indices", dtype=Type.i32)
    block_count = parameter([1, 12, 2], name="block_count", dtype=Type.i32)
    node = ov.sparse_attention(query, key, value, block_indices, block_count,
                               key_block_size=128, query_block_size=128, causal=False)

    assert node.get_type_name() == "SparseAttention"
    assert node.get_output_partial_shape(0) == PartialShape([1, 12, 256, 128])


def test_sparse_attention_skipped_optional_input():
    query = parameter([2, 24, 5, 64], name="query", dtype=Type.f32)
    key = parameter([2, 2, 12, 64], name="key", dtype=Type.f32)
    value = parameter([2, 2, 12, 32], name="value", dtype=Type.f32)
    block_indices = parameter([2, 1, 5, 4], name="block_indices", dtype=Type.i32)
    block_count = parameter([2, 1, 5], name="block_count", dtype=Type.i32)
    sinks = parameter([24], name="sinks", dtype=Type.f32)

    with pytest.raises(ValueError, match="positional"):
        ov.sparse_attention(query, key, value, block_indices, block_count,
                            sinks=sinks, key_block_size=4)


def test_sparse_attention_indexer():
    index_query = parameter([2, 4, 7, 128], name="index_query", dtype=Type.f32)
    block_index_keys = parameter([2, 2500, 128], name="block_index_keys", dtype=Type.f32)
    key_length = parameter([2], name="key_length", dtype=Type.i32)
    node = ov.sparse_attention_indexer(index_query, block_index_keys, key_length,
                                       key_block_size=4, token_budget=2048)

    assert node.get_type_name() == "SparseAttentionIndexer"
    assert node.get_output_size() == 2
    assert node.get_output_element_type(0) == Type.i32
    assert node.get_output_element_type(1) == Type.i32
    assert node.get_output_partial_shape(0) == PartialShape([2, 1, 7, 512])
    assert node.get_output_partial_shape(1) == PartialShape([2, 1, 7])


def test_sparse_attention_indexer_with_weights_and_selection_heads():
    index_query = parameter([1, 8, 7, 128], name="index_query", dtype=Type.f32)
    block_index_keys = parameter([1, 100, 128], name="block_index_keys", dtype=Type.f32)
    key_length = parameter([1], name="key_length", dtype=Type.i64)
    index_weights = parameter([1, 7, 8], name="index_weights", dtype=Type.f32)
    node = ov.sparse_attention_indexer(index_query, block_index_keys, key_length, index_weights,
                                       key_block_size=4, token_budget=64, num_selection_heads=2)

    assert node.get_type_name() == "SparseAttentionIndexer"
    assert len(node.inputs()) == 4
    assert node.get_output_partial_shape(0) == PartialShape([1, 2, 7, 16])
    assert node.get_output_partial_shape(1) == PartialShape([1, 2, 7])


def test_qsa_prepare():
    raw_index_keys = parameter([2, 8, 128], name="raw_index_keys", dtype=Type.f32)
    past_block_index_keys = parameter([2, -1, 128], name="past_block_index_keys", dtype=Type.f32)
    key_length = parameter([2], name="key_length", dtype=Type.i32)
    rope_cos = parameter([2, 8, 64], name="rope_cos", dtype=Type.f32)
    rope_sin = parameter([2, 8, 64], name="rope_sin", dtype=Type.f32)
    k_norm_weight = parameter([128], name="k_norm_weight", dtype=Type.f32)
    past_incomplete_raw_keys = parameter([2, 3, 128], name="past_incomplete_raw_keys", dtype=Type.f32)
    past_incomplete_rope = parameter([2, 128], name="past_incomplete_rope", dtype=Type.f32)
    node = ov.qsa_prepare(raw_index_keys, past_block_index_keys, key_length, rope_cos, rope_sin,
                          k_norm_weight, past_incomplete_raw_keys, past_incomplete_rope,
                          key_block_size=4)

    assert node.get_type_name() == "QSAPrepare"
    assert node.get_output_size() == 3
    assert node.get_output_element_type(0) == Type.f32
    # The number of complete blocks follows from the key_length values, unknown here.
    assert node.get_output_partial_shape(0) == PartialShape([2, -1, 128])
    assert node.get_output_partial_shape(1) == PartialShape([2, 3, 128])
    assert node.get_output_partial_shape(2) == PartialShape([2, 128])


def test_qsa_prepare_without_rotation():
    raw_index_keys = parameter([1, 8, 128], name="raw_index_keys", dtype=Type.f32)
    past_block_index_keys = parameter([1, 0, 128], name="past_block_index_keys", dtype=Type.f32)
    key_length = parameter([1], name="key_length", dtype=Type.i64)
    empty = constant(np.empty(0, dtype=np.float32))
    k_norm_weight = parameter([128], name="k_norm_weight", dtype=Type.f32)
    past_incomplete_raw_keys = parameter([1, 3, 128], name="past_incomplete_raw_keys", dtype=Type.f32)
    node = ov.qsa_prepare(raw_index_keys, past_block_index_keys, key_length, empty, empty,
                          k_norm_weight, past_incomplete_raw_keys, empty,
                          key_block_size=4, epsilon=1e-5)

    assert node.get_type_name() == "QSAPrepare"
    assert node.get_output_partial_shape(2) == PartialShape([1, 0])
