// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/paged_attention.hpp"

#include "dimension_util.hpp"
#include "itt.hpp"
#include "openvino/op/op.hpp"

namespace {

using namespace ov;
using namespace ov::element;

// Validates input rank and type for a node input.
// We consider that dynamic rank/type are always valid case.
// Empty {} means any rank/type
inline void input_check(ov::Node* node,
                        size_t idx,
                        const std::string& input_name,
                        const std::set<int64_t>& allowed_ranks,
                        std::set<Type> allowed_types) {
    using namespace ov;
    using namespace ov::element;

    const auto& pshape = node->get_input_partial_shape(idx);
    const auto& etype = node->get_input_element_type(idx);

    auto rank_check = [&](const PartialShape& ps) {
        return ps.rank().is_dynamic() || allowed_ranks.empty() || allowed_ranks.count(ps.rank().get_length());
    };

    auto type_check = [&](const Type& type) {
        return type.is_dynamic() || allowed_types.empty() || allowed_types.count(type);
    };

    NODE_VALIDATION_CHECK(
        node,
        rank_check(pshape),
        "Rank of `",
        input_name,
        "` input should be in [dynamic, ",
        [=]() {
            std::ostringstream oss;
            bool first = true;
            for (auto r : allowed_ranks) {
                if (!first)
                    oss << ", ";
                oss << r;
                first = false;
            }
            return oss.str();
        }(),
        "] list, but it is ",
        pshape.rank(),
        ".");

    NODE_VALIDATION_CHECK(
        node,
        type_check(etype),
        "Element type of `",
        input_name,
        "` input should be in [dynamic, ",
        [=]() {
            std::ostringstream oss;
            bool first = true;
            for (auto t : allowed_types) {
                if (!first)
                    oss << ", ";
                oss << t;
                first = false;
            }
            return oss.str();
        }(),
        "] list, but it is ",
        etype,
        ".");
}

std::set<Type> get_real_types() {
    std::set<Type> real_types;
    for (const auto& type : element::Type::get_known_types()) {
        if (type->is_real()) {
            real_types.insert(*type);
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
                          get_input_size() == 13 || get_input_size() == 16,
                          "PagedAttensionExtension expects 13 or 16 inputs, but it has ",
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
    input_check(this, 9, "scale", {0}, get_real_types());
    input_check(this, 10, "sliding_window", {0}, {element::i32});
    input_check(this, 11, "alibi_slopes", {1}, get_real_types());
    input_check(this, 12, "max_context_len", {0}, {element::i32});

    if (get_input_size() == 16) {
        input_check(this, 13, "rotated_block_indices", {1}, {element::i32});
        input_check(this, 14, "rotation_deltas", {2}, {element::i32});
        input_check(this, 15, "rotation_trig_lut", {2}, {element::f16, element::f32});
    }

    // value head_size may be not same with key
    auto out_ps = get_input_partial_shape(0);
    const auto& key_ps = get_input_partial_shape(1);
    const auto& value_ps = get_input_partial_shape(2);
    if (out_ps.rank().is_static()) {
        if (key_ps.rank().is_static() && value_ps.rank().is_static() && key_ps[1].is_static()) {
            // The dim of out_ps[1] should be `num_heads * v_head_size`, it can be got from:
            // because:
            //   q: query_ps[1] = num_heads * head_size
            //   k: key_ps[1] = num_kv_heads * head_size
            //   v: value_ps[1] = num_kv_heads * v_head_size
            // therefore:
            //   q * v / k = (num_heads * head_size) * (num_kv_heads * v_head_size) /
            //               (num_kv_heads * head_size) = num_heads * v_head_size
            out_ps[1] = out_ps[1] * value_ps[1] / key_ps[1].get_length();
            NODE_VALIDATION_CHECK(this,
                                  !ov::util::dim::is_empty(out_ps[1]),
                                  "The last dimension of output should not be empty.");
        } else {
            out_ps[1] = Dimension::dynamic();
        }
    }
    if (m_output_type[0].is_dynamic()) {
        set_output_type(0, get_input_element_type(0), out_ps);
    } else {
        set_output_type(0, m_output_type[0], out_ps);
    }

    if (m_output_type[1].is_dynamic()) {
        set_output_type(1, get_input_element_type(0), {Dimension::dynamic()});
    } else {
        set_output_type(1, m_output_type[1], {Dimension::dynamic()});
    }
}

std::shared_ptr<ov::Node> PagedAttentionExtension::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    return std::make_shared<PagedAttentionExtension>(new_args);
}

void PagedAttentionExtension::set_out_type(int index, const ov::element::Type& output_type) {
    OPENVINO_ASSERT(index < 2, "Output index should be 0 or 1, but got " + std::to_string(index));
    m_output_type[index] = output_type;
}

}  // namespace op
}  // namespace ov
