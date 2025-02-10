# -*- coding: utf-8 -*-
# Copyright (C) 2018-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest

from openvino import Type
from openvino import opset15


def test_embedding_bag_offsets_15():
    emb_table = opset15.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = opset15.parameter([4], name="indices", dtype=np.int64)
    offsets = opset15.parameter([3], name="offsets", dtype=np.int64)

    node = opset15.embedding_bag_offsets(emb_table, indices, offsets)

    assert node.get_type_name() == "EmbeddingBagOffsets"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_attributes()["reduction"] == "sum"


def test_embedding_bag_offsets_15_default_index():
    emb_table = opset15.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = opset15.parameter([4], name="indices", dtype=np.int64)
    offsets = opset15.parameter([3], name="offsets", dtype=np.int64)
    default_index = opset15.parameter([], name="default_index", dtype=np.int64)

    node = opset15.embedding_bag_offsets(emb_table, indices, offsets, default_index, reduction="MeAn")

    assert node.get_type_name() == "EmbeddingBagOffsets"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_attributes()["reduction"] == "mean"


def test_embedding_bag_offsets_15_per_sample_weights():
    emb_table = opset15.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = opset15.parameter([4], name="indices", dtype=np.int64)
    offsets = opset15.parameter([3], name="offsets", dtype=np.int64)
    per_sample_weights = opset15.parameter([4], name="per_sample_weights", dtype=np.float32)

    node = opset15.embedding_bag_offsets(emb_table, indices, offsets, per_sample_weights=per_sample_weights, reduction="SUM")

    assert node.get_type_name() == "EmbeddingBagOffsets"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_attributes()["reduction"] == "sum"


def test_embedding_bag_offsets_15_default_index_per_sample_weights():
    emb_table = opset15.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = opset15.parameter([4], name="indices", dtype=np.int64)
    offsets = opset15.parameter([3], name="offsets", dtype=np.int64)
    default_index = opset15.parameter([], name="default_index", dtype=np.int64)
    per_sample_weights = opset15.parameter([4], name="per_sample_weights", dtype=np.float32)

    node = opset15.embedding_bag_offsets(emb_table, indices, offsets, default_index, per_sample_weights, "sum")

    assert node.get_type_name() == "EmbeddingBagOffsets"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_attributes()["reduction"] == "sum"


def test_embedding_bag_packed_15():
    emb_table = opset15.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = opset15.parameter([3, 3], name="indices", dtype=np.int64)

    node = opset15.embedding_bag_packed(emb_table, indices, reduction="mEaN")

    assert node.get_type_name() == "EmbeddingBagPacked"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_attributes()["reduction"] == "mean"


def test_embedding_bag_packed_15_per_sample_weights():
    emb_table = opset15.parameter([5, 2], name="emb_table", dtype=np.float32)
    indices = opset15.parameter([3, 3], name="indices", dtype=np.int64)
    per_sample_weights = opset15.parameter([3, 3], name="per_sample_weights", dtype=np.float32)

    node = opset15.embedding_bag_packed(emb_table, indices, per_sample_weights)

    assert node.get_type_name() == "EmbeddingBagPacked"
    assert node.get_output_size() == 1
    assert list(node.get_output_shape(0)) == [3, 2]
    assert node.get_output_element_type(0) == Type.f32
    assert node.get_attributes()["reduction"] == "sum"
