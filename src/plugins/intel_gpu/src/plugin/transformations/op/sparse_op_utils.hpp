// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <initializer_list>
#include <string>

#include "intel_gpu/op/sparse_kv_quant.hpp"
#include "openvino/core/attribute_visitor.hpp"
#include "openvino/core/node.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/validation_util.hpp"

namespace ov::intel_gpu::op::sparse_op_util {

/// Checks that the input `idx` of `node` is compatible with `expected`.
inline void check_shape(const ov::Node* node, size_t idx, const char* name, const ov::PartialShape& expected) {
    const auto& ps = node->get_input_partial_shape(idx);
    NODE_VALIDATION_CHECK(node, ps.compatible(expected), node->get_type_name(), " expects ", name, " shaped ", expected, ", got ", ps);
}

/// Returns the dimension at `axis`, or a dynamic dimension when the rank is dynamic or too small.
inline ov::Dimension dim_at(const ov::PartialShape& ps, size_t axis) {
    return ps.rank().is_static() && ps.size() > axis ? ps[axis] : ov::Dimension::dynamic();
}

/// Checks that the inputs `indices` of `node` share one floating-point (or dynamic) element type.
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

/// Config rules shared by SparseSDPA::Config and SparsePA::Config.
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

/// Serializes the key/value quantization attributes shared by SparseSDPA::Config and SparsePA::Config.
inline void visit_kv_quant(ov::AttributeVisitor& visitor, const std::string& prefix, KVQuant& quant) {
    auto mode = static_cast<int32_t>(quant.mode);
    auto bits = static_cast<int32_t>(quant.bits);
    auto group_size = static_cast<int32_t>(quant.group_size);
    visitor.on_attribute(prefix + "quant_mode", mode);
    visitor.on_attribute(prefix + "quant_bits", bits);
    visitor.on_attribute(prefix + "quant_symmetric", quant.symmetric);
    visitor.on_attribute(prefix + "quant_group_size", group_size);
    quant.mode = static_cast<KVQuant::Mode>(mode);
    quant.bits = static_cast<uint8_t>(bits);
    quant.group_size = static_cast<uint16_t>(group_size);
}

}  // namespace ov::intel_gpu::op::sparse_op_util
