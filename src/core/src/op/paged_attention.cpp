// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "paged_attention_shape_inference.hpp"

namespace {

// Validates input rank and type for a node input.
// We consider that dynamic rank/type are always valid case.
// Empty {} means any rank/type
inline void input_check(const ov::Node* node,
                        size_t idx,
                        const std::string_view input_name,
                        std::initializer_list<ov::Rank>&& allowed_ranks,
                        const std::vector<ov::element::Type>& allowed_types) {
    using namespace ov;
    using namespace ov::util;
    using namespace ov::element;

    const auto& rank = node->get_input_partial_shape(idx).rank();
    const auto& tp = node->get_input_element_type(idx);

    auto rank_check = [&](const Rank& rank) {
        return rank.is_dynamic() || empty(allowed_ranks) || is_rank_compatible_any_of(rank.get_length(), allowed_ranks);
    };

    auto type_check = [&](const Type& type) {
        auto it = std::find(allowed_types.begin(), allowed_types.end(), type);
        return type.is_dynamic() || allowed_types.empty() || it != allowed_types.end();
    };

    NODE_VALIDATION_CHECK(node,
                          rank_check(rank),
                          "Rank of `",
                          input_name,
                          "` input should be in [dynamic, ",
                          join(allowed_ranks),
                          "] list, but it is ",
                          rank,
                          ".");

    NODE_VALIDATION_CHECK(node,
                          type_check(tp),
                          "Element type of `",
                          input_name,
                          "` input should be in [dynamic, ",
                          join(allowed_types),
                          "] list, but it is ",
                          tp,
                          ".");
}

std::vector<ov::element::Type> get_real_types() {
    std::vector<ov::element::Type> real_types;
    for (const auto& type : ov::element::Type::get_known_types()) {
        if (type->is_real()) {
            real_types.push_back(*type);
        }
    }
    return real_types;
}

}  // namespace

namespace ov {
namespace op {
PagedAttentionExtension::PagedAttentionExtension(const ov::OutputVector& args) : ov::op::Op(args) {
    constructor_validate_and_infer_types();
}

void PagedAttentionExtension::validate_and_infer_types() {
    OV_OP_SCOPE(PagedAttentionExtension_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 25,
                          "PagedAttensionExtension expects 25 inputs, but it has ",
                          get_input_size());

    // format: Node*, input_idx, name, {rank_list}, {type_list}
    input_check(this, 0, "query", {2}, {});
    input_check(this, 1, "key", {2}, {});
    input_check(this, 2, "value", {2}, {});
    input_check(this, 3, "key_cache", {2, 3, 4, 5}, {});
    input_check(this, 4, "value_cache", {2, 3, 4, 5}, {});
    input_check(this, 5, "past_lens", {1}, {element::i32});
    input_check(this, 6, "subsequence_begins", {1}, {element::i32});
    input_check(this, 7, "block_indices", {1}, {element::i32});
    input_check(this, 8, "block_indices_begins", {1}, {element::i32});
    input_check(this, 9, "scale", {0, 1}, get_real_types());
    input_check(this, 10, "sliding_window", {0}, {element::i32});
    input_check(this, 11, "alibi_slopes", {1}, get_real_types());
    input_check(this, 12, "max_context_len", {0}, {element::i32});
    input_check(this, 13, "score_aggregation_window", {0, 1}, {element::i32});
    input_check(this, 14, "rotated_block_indices", {1}, {element::i32});
    input_check(this, 15, "rotation_deltas", {1, 2}, {element::i32});
    input_check(this, 16, "rotation_trig_lut", {1, 2}, {element::f16, element::f32});
    input_check(this, 17, "xattention_threshold", {0, 1}, {element::f16, element::f32});
    input_check(this, 18, "xattention_block_size", {0}, {element::i32});
    input_check(this, 19, "xattention_stride", {0}, {element::i32});
    input_check(this, 20, "sinks", {1, 4}, {});
    input_check(this, 21, "adaptive_rkv_start_size", {0}, {element::i32});
    input_check(this, 22, "adaptive_rkv_evictable_sizes", {1}, {element::i32});
    input_check(this, 23, "adaptive_rkv_diversity_block_set_indices", {1}, {element::i32});
    input_check(this, 24, "adaptive_rkv_diversity_block_set_indices_begins", {1}, {element::i32});

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    // Honour m_output_type overrides (e.g. the CPU plugin pins output 1 to f32
    // for f16 inference via set_out_type). Fall back to the query element type
    // only when no override has been set (m_output_type is element::dynamic)
    for (int i = 0; i < 3; ++i) {
        const auto et = m_output_type[i].is_dynamic() ? get_input_element_type(0) : m_output_type[i];
        set_output_type(i, et, output_shapes[i]);
    }
}

std::shared_ptr<ov::Node> PagedAttentionExtension::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedAttentionExtension_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedAttentionExtension>(new_args);
}

const ov::element::Type PagedAttentionExtension::get_out_type(int index) const {
    OPENVINO_ASSERT(index < 3, "Output index should be 0, 1 or 2, but got " + std::to_string(index));
    return m_output_type[index];
}

void PagedAttentionExtension::set_out_type(int index, const ov::element::Type& output_type) {
    OPENVINO_ASSERT(index < 3, "Output index should be 0, 1 or 2, but got " + std::to_string(index));
    m_output_type[index] = output_type;
}

}  // namespace op
}  // namespace ov
