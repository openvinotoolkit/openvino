// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_sparse_attention.hpp"

#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/util/common_util.hpp"
#include "openvino/util/container_util.hpp"
#include "paged_sparse_attention_shape_inference.hpp"

namespace ov::op::internal {

PagedSparseAttention::PagedSparseAttention(const OutputVector& inputs, int64_t key_block_size)
    : Op(inputs),
      m_key_block_size(key_block_size) {
    constructor_validate_and_infer_types();
}

void PagedSparseAttention::validate_and_infer_types() {
    OV_OP_SCOPE(PagedSparseAttention_validate_and_infer_types);
    NODE_VALIDATION_CHECK(this,
                          m_key_block_size > 0,
                          "key_block_size must be positive, but got ",
                          m_key_block_size,
                          ".");
    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 13,
                          "PagedSparseAttention expects 13 inputs, but got ",
                          get_input_size(),
                          ".");

    auto data_type = get_input_element_type(0);
    const auto& key_type = get_input_element_type(1);
    const auto& value_type = get_input_element_type(2);
    const auto& key_cache_type = get_input_element_type(3);
    const auto& value_cache_type = get_input_element_type(4);
    const auto& scale_type = get_input_element_type(11);
    const auto& sinks_type = get_input_element_type(12);

    const auto data_types_match = element::Type::merge(data_type, data_type, key_type) &&
                                  element::Type::merge(data_type, data_type, value_type) &&
                                  element::Type::merge(data_type, data_type, scale_type);
    NODE_VALIDATION_CHECK(this, data_types_match);
    NODE_VALIDATION_CHECK(this,
                          data_type.is_dynamic() || data_type.is_real(),
                          "Query, key, value and scale must be floating-point.");
    NODE_VALIDATION_CHECK(this,
                          sinks_type.is_dynamic() || sinks_type.is_real(),
                          "The sinks element type must be floating-point, but got ",
                          sinks_type,
                          ".");

    constexpr auto common_cache_types =
        ov::util::make_array(element::dynamic, element::u4, element::u8, element::f16, element::f32, element::bf16);
    NODE_VALIDATION_CHECK(this,
                          ov::util::contains(common_cache_types, key_cache_type) || key_cache_type == element::i8);
    NODE_VALIDATION_CHECK(this, ov::util::contains(common_cache_types, value_cache_type));

    for (size_t input_index = 5; input_index <= 10; ++input_index) {
        const auto& input_type = get_input_element_type(input_index);
        NODE_VALIDATION_CHECK(this,
                              input_type.compatible(element::i32),
                              "Input ",
                              input_index,
                              " must have element type i32, but got ",
                              input_type,
                              ".");
    }

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    set_output_type(0, data_type, output_shapes[0]);
}

bool PagedSparseAttention::visit_attributes(AttributeVisitor& visitor) {
    OV_OP_SCOPE(PagedSparseAttention_visit_attributes);
    visitor.on_attribute("key_block_size", m_key_block_size);
    return true;
}

std::shared_ptr<Node> PagedSparseAttention::clone_with_new_inputs(const OutputVector& new_args) const {
    OV_OP_SCOPE(PagedSparseAttention_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedSparseAttention>(new_args, m_key_block_size);
}

}  // namespace ov::op::internal
