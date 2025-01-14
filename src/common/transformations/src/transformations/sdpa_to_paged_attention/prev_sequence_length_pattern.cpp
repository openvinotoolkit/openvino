// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sdpa_to_paged_attention/prev_sequence_length_pattern.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

using namespace ov::op;

ov::pass::PrevSequenceLengthPattern::PrevSequenceLengthPattern(const std::shared_ptr<ov::Node>& unsqueezed_input_ids,
                                                               const std::shared_ptr<ov::Node>& max_context_len,
                                                               const std::shared_ptr<ov::Node>& position_ids) {
    MATCHER_SCOPE(PrevSequenceLengthPattern);
    // The transformation addresses two cases that look similar: (1) previous sequence length, (2) batch size in
    // kv-cache state In first case it should replace it by prev_max_seq_len. For the second case, connect to batch_dim.

    auto kv_past = pattern::wrap_type<v6::ReadValue>({pattern::any_input()});
    auto kv_gather = pattern::wrap_type<v8::Gather>({kv_past, pattern::any_input(), pattern::any_input()});
    auto kv_shape = pattern::wrap_type<v3::ShapeOf>({kv_gather});
    auto seq = pattern::wrap_type<v8::Gather>({kv_shape, pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        // TODO: Check that seq has axis that really takes sequence len but not any other dimension -- use symbolics or
        // look at the constant input
        // Detect the case by taking initialization expression for ReadValue and compare it with the second gather index
        const auto& pattern_map = m.get_pattern_value_map();
        auto gather = m.get_match_root();
        auto gather_index = ov::util::get_constant_from_source(gather->input_value(1));
        if (!gather_index) {
            return false;  // cannot detect axis
        }
        auto axis = gather_index->cast_vector<int64_t>().at(0);
        auto kv_init_shape = pattern_map.at(kv_past).get_node()->get_input_partial_shape(0);
        auto target_type = gather->get_output_element_type(0);
        std::shared_ptr<ov::Node> replacement;
        if (kv_init_shape[axis].is_static() && kv_init_shape[axis].get_length() == 0) {
            auto cur_seq_len = std::make_shared<v8::Gather>(std::make_shared<v3::ShapeOf>(unsqueezed_input_ids),
                                                            v0::Constant::create(element::i64, Shape{}, {1}),
                                                            v0::Constant::create(element::i64, Shape{}, {0}));
            auto cur_seq_len_i32 = std::make_shared<v0::Convert>(cur_seq_len, element::i32);
            auto prev_max_seq_len = std::make_shared<v1::Subtract>(max_context_len, cur_seq_len_i32);
            replacement = prev_max_seq_len;
        } else {
            // it is not always required, so will be disposed if not needed
            auto batch_dim = std::make_shared<v3::ShapeOf>(position_ids);

            // assumption that any other axis should point to batch dimension, precise reasoning is too complex
            // TODO: provide more reliable check
            replacement = batch_dim;
        }
        if (replacement->get_output_element_type(0) != target_type) {
            replacement = std::make_shared<v0::Convert>(replacement, target_type);
        }
        auto required_shape = gather->get_output_partial_shape(0);
        if (replacement->get_output_partial_shape(0) != required_shape && required_shape.rank().is_static()) {
            replacement = op::util::reshapeTo(replacement, Shape(required_shape.rank().get_length(), 1));
        }
        replace_node(gather, replacement);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(seq, matcher_name);
    register_matcher(m, callback);
}
