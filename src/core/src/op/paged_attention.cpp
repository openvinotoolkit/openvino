// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/op.hpp"
#include "openvino/util/common_util.hpp"
#include "paged_attention_shape_inference.hpp"

namespace {

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

PagedAttentionExtension::PagedAttentionExtension(const ov::OutputVector& args, bool write_kv_cache)
    : ov::op::Op(args),
      m_write_kv_cache(write_kv_cache) {
    constructor_validate_and_infer_types();
}

void PagedAttentionExtension::validate_and_infer_types() {
    OV_OP_SCOPE(PagedAttentionExtension_validate_and_infer_types);

    NODE_VALIDATION_CHECK(this,
                          get_input_size() == 28,
                          "PagedAttensionExtension expects 28 inputs, but it has ",
                          get_input_size());

    // format: Node*, input_idx, name, {rank_list}, {type_list}
    ov::util::validate_input_rank_and_type(this, 0, "query", {2}, {});
    ov::util::validate_input_rank_and_type(this, 1, "key", {2}, {});
    ov::util::validate_input_rank_and_type(this, 2, "value", {2}, {});
    ov::util::validate_input_rank_and_type(this, 3, "key_cache", {2, 3, 4, 5}, {});
    ov::util::validate_input_rank_and_type(this, 4, "value_cache", {2, 3, 4, 5}, {});
    ov::util::validate_input_rank_and_type(this, 5, "past_lens", {1}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 6, "subsequence_begins", {1}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 7, "block_indices", {1}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 8, "block_indices_begins", {1}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 9, "scale", {0}, get_real_types());
    ov::util::validate_input_rank_and_type(this, 10, "sliding_window", {0}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 11, "alibi_slopes", {1}, get_real_types());
    ov::util::validate_input_rank_and_type(this, 12, "max_context_len", {0}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 13, "score_aggregation_window", {0, 1}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 14, "rotated_block_indices", {1}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 15, "rotation_deltas", {1, 2}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 16, "rotation_trig_lut", {1, 2}, {element::f16, element::f32});
    ov::util::validate_input_rank_and_type(this, 17, "xattention_threshold", {1}, {element::f16, element::f32});
    ov::util::validate_input_rank_and_type(this, 18, "xattention_block_size", {0}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 19, "xattention_stride", {0}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 20, "sinks", {1, 4}, {});

    ov::util::validate_input_rank_and_type(this, 21, "adaptive_rkv_start_size", {0}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 22, "adaptive_rkv_evictable_sizes", {1}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 23, "adaptive_rkv_diversity_block_set_indices", {1}, {element::i32});
    ov::util::validate_input_rank_and_type(this,
                                           24,
                                           "adaptive_rkv_diversity_block_set_indices_begins",
                                           {1},
                                           {element::i32});
    ov::util::validate_input_rank_and_type(this, 25, "token_type_ids", {1, 2}, {element::i32});
    ov::util::validate_input_rank_and_type(this, 26, "qq_bias", {1}, {element::u8});
    ov::util::validate_input_rank_and_type(this, 27, "qq_bias_begins", {1}, {element::i32});

    const auto input_shapes = ov::util::get_node_input_partial_shapes(*this);
    const auto output_shapes = shape_infer(this, input_shapes);
    // Use m_output_type overrides when set, otherwise fall back to query type
    for (int i = 0; i < 3; ++i) {
        const auto et = m_output_type[i].is_dynamic() ? get_input_element_type(0) : m_output_type[i];
        set_output_type(i, et, output_shapes[i]);
    }
}

std::shared_ptr<ov::Node> PagedAttentionExtension::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    OV_OP_SCOPE(PagedAttentionExtension_clone_with_new_inputs);
    check_new_args_count(this, new_args);
    return std::make_shared<PagedAttentionExtension>(new_args, m_write_kv_cache);
}

bool PagedAttentionExtension::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("write_kv_cache", m_write_kv_cache);
    return true;
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
