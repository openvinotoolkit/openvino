// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

// Validation helpers shared by the QSA indexer / sparse-attention ops (GPU plugin).

#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>

#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov::intel_gpu::op::qsa_util {

// The kernels read every integer input as int32_t.
inline void check_i32_input(const ov::Node* node, size_t idx, const char* name, int64_t rank) {
    const auto& et = node->get_input_element_type(idx);
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node, et.is_dynamic() || et == element::i32, node->get_type_name(), " expects an i32 ", name, " input, got ", et);
    NODE_VALIDATION_CHECK(node, ps.rank().compatible(rank), node->get_type_name(), " expects a rank-", rank, " ", name, " input, got ", ps);
}

inline void check_shape(const ov::Node* node, size_t idx, const char* name, const ov::PartialShape& expected) {
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node, ps.compatible(expected), node->get_type_name(), " expects ", name, " shaped ", expected, ", got ", ps);
}

inline void check_float_inputs(const ov::Node* node, std::initializer_list<size_t> indices) {
    element::Type et = element::dynamic;
    for (auto idx : indices) {
        const auto& in_et = node->get_input_element_type(idx);
        NODE_VALIDATION_CHECK(node,
                              element::Type::merge(et, et, in_et),
                              node->get_type_name(),
                              " expects all floating-point inputs to share one element type; input ",
                              idx,
                              " is ",
                              in_et);
    }
    NODE_VALIDATION_CHECK(node, et.is_dynamic() || et.is_real(), node->get_type_name(), " expects floating-point inputs, got ", et);
}

inline ov::Dimension dim_at(const ov::PartialShape& ps, size_t axis) {
    return ps.rank().is_static() && ps.size() > axis ? ps[axis] : ov::Dimension::dynamic();
}

/// Config rules common to SparseSDPA::Config and SparsePA::Config.
template <class Config>
bool validate_sparse_attention_config(const Config& c, std::string& reason) {
    auto fail = [&reason](const char* msg) {
        reason = msg;
        return false;
    };
    if (c.num_heads == 0 || c.num_kv_heads == 0 || c.num_heads % c.num_kv_heads != 0) {
        return fail("num_heads must be a positive multiple of num_kv_heads");
    }
    if (c.num_selection_heads != 1 && c.num_selection_heads != c.num_kv_heads && c.num_selection_heads != c.num_heads) {
        return fail("num_selection_heads must be 1, num_kv_heads, or num_heads");
    }
    if (c.k_head_size == 0 || c.k_head_size % 16 != 0 || c.v_head_size == 0 || c.v_head_size % 16 != 0) {
        return fail("k_head_size / v_head_size must be positive multiples of 16");
    }
    if (c.key_block_size == 0) {
        return fail("key_block_size must be positive");
    }
    if (c.query_block_size != 1) {
        return fail("query_block_size != 1 is not implemented in this pass");
    }
    return true;
}

template <class Config>
void visit_sparse_attention_config(ov::AttributeVisitor& visitor, Config& c) {
    visitor.on_attribute("num_heads", c.num_heads);
    visitor.on_attribute("num_kv_heads", c.num_kv_heads);
    visitor.on_attribute("num_selection_heads", c.num_selection_heads);
    visitor.on_attribute("k_head_size", c.k_head_size);
    visitor.on_attribute("v_head_size", c.v_head_size);
    visitor.on_attribute("key_block_size", c.key_block_size);
    visitor.on_attribute("query_block_size", c.query_block_size);
    visitor.on_attribute("causal", c.causal);
}

}  // namespace ov::intel_gpu::op::qsa_util
