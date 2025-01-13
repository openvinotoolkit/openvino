// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sdpa_to_paged_attention/total_sequence_length_pattern.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;
using namespace ov::pass::pattern;

namespace {

void align_replacement(std::shared_ptr<ov::Node>& replacement,
                       const ov::PartialShape& required_shape,
                       ov::element::Type target_type) {
    if (replacement->get_output_element_type(0) != target_type) {
        replacement = std::make_shared<v0::Convert>(replacement, target_type);
    }

    if (replacement->get_output_partial_shape(0) != required_shape && required_shape.rank().is_static()) {
        replacement = ov::op::util::reshapeTo(replacement, ov::Shape(required_shape.rank().get_length(), 1));
    }
}

}  // namespace

ov::pass::TotalSequenceLengthPattern::TotalSequenceLengthPattern(
    const std::shared_ptr<ov::op::v0::Parameter>& max_context_len) {
    MATCHER_SCOPE(TotalSequenceLengthPattern);

    auto kv_past = wrap_type<v6::ReadValue>({any_input()});
    auto kv_gather = wrap_type<v8::Gather>({kv_past, any_input(), any_input()});
    auto kv_current = any_input();
    auto kv_concat = wrap_type<v0::Concat>({kv_gather, kv_current});
    auto kv_shape = wrap_type<v3::ShapeOf>({kv_concat});
    auto gather_idx_label = wrap_type<v0::Constant>();
    auto seq = wrap_type<v8::Gather>({kv_shape, gather_idx_label, any_input()});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        // TODO: Check that seq has axis that really takes sequence len but not any other dimension --
        //  use symbolic infra or look at the constant input
        const auto& pattern_map = m.get_pattern_value_map();

        auto concat = ov::as_type_ptr<v0::Concat>(pattern_map.at(kv_concat).get_node_shared_ptr());
        auto gather = pattern_map.at(seq).get_node_shared_ptr();
        auto gather_idx = ov::as_type_ptr<v0::Constant>(pattern_map.at(gather_idx_label).get_node_shared_ptr());

        if (!concat || !gather || !gather_idx || !gather_idx) {
            return false;
        }

        auto gather_idx_data = gather_idx->cast_vector<int64_t>();

        if (gather_idx_data.size() != 1) {
            return false;
        }

        int64_t gather_idx_to_compare = gather_idx_data[0];

        if (gather_idx_data[0] < 0) {
            if (gather->input(0).get_partial_shape().is_static()) {
                const auto& gather_data_shape = gather->input(0).get_shape();
                gather_idx_to_compare = ov::util::normalize(gather_idx_data[0], gather_data_shape[0]);
            } else {
                return false;
            }
        }

        std::shared_ptr<Node> replacement = max_context_len;

        int64_t concat_axis_to_compare = concat->get_axis();
        if (concat_axis_to_compare < 0) {
            // If it's dynamic, leave it negative as we cannot take dynamic
            // dimension here so the next comparison would fail
            if (concat->get_output_partial_shape(0).is_static()) {
                const auto& concat_output_shape = concat->output(0).get_partial_shape();
                concat_axis_to_compare =
                    ov::util::normalize(concat_axis_to_compare, concat_output_shape.rank().get_length());
            }
        }

        if (concat_axis_to_compare == gather_idx_to_compare) {
            auto target_type = gather->get_output_element_type(0);
            auto required_shape = gather->get_output_partial_shape(0);
            align_replacement(replacement, required_shape, target_type);
        } else {
            // TODO: change in the future when we start supporting dynamic shapes here
            replacement = ov::util::get_constant_from_source(gather->output(0));
            OPENVINO_ASSERT(replacement,
                            "TotalSequenceLengthPattern transformation failed to determine the dimension value after ",
                            "the Gather operation. Most probably, the required dimension is dynamic: ",
                            concat);
        }

        replace_node(gather, replacement);
        return true;
    };

    auto m = std::make_shared<Matcher>(seq, matcher_name);
    register_matcher(m, callback);
}

ov::pass::TotalSequenceLengthPatternQwen::TotalSequenceLengthPatternQwen(
    const std::shared_ptr<ov::op::v0::Parameter>& max_context_len) {
    MATCHER_SCOPE(TotalSequenceLengthPatternQwen);

    auto p_input_ids = wrap_type<v0::Parameter>();
    auto p_unsqueeze = wrap_type<v0::Unsqueeze>({p_input_ids, any_input()});
    auto p_opt_reshape_1 = optional<v1::Reshape>({p_unsqueeze, any_input()});
    auto p_opt_convert_1 = optional<v0::Convert>(p_opt_reshape_1);
    auto p_kv_shape_current = wrap_type<v3::ShapeOf>({p_opt_convert_1});
    auto p_seq_current = wrap_type<v8::Gather>({p_kv_shape_current, any_input(), any_input()});
    auto p_opt_convert_2 = optional<v0::Convert>(p_seq_current);

    auto p_max_context_len = wrap_type<v0::Parameter>();
    auto p_prev_max_seq_len = wrap_type<v1::Subtract>({p_max_context_len, any_input()});
    auto p_opt_convert_3 = optional<v0::Convert>(p_prev_max_seq_len);
    auto p_opt_reshape_2 = optional<v1::Reshape>({p_opt_convert_3, any_input()});
    auto p_total_seq = wrap_type<v1::Add>({p_opt_convert_2, p_opt_reshape_2});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto total_seq = pattern_map.at(p_total_seq).get_node_shared_ptr();
        std::shared_ptr<Node> replacement = max_context_len;

        auto target_type = total_seq->get_output_element_type(0);
        auto required_shape = total_seq->get_output_partial_shape(0);
        align_replacement(replacement, required_shape, target_type);

        replace_node(total_seq, replacement);
        return true;
    };

    auto m = std::make_shared<Matcher>(p_total_seq, matcher_name);
    register_matcher(m, callback);
}
