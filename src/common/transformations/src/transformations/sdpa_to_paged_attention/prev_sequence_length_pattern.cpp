// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sdpa_to_paged_attention/prev_sequence_length_pattern.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/core/validation_util.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

ov::pass::PrevSequenceLengthPattern::PrevSequenceLengthPattern(
    const std::shared_ptr<ov::op::v1::Subtract>& prev_max_seq_len,
    std::shared_ptr<ov::Node> batch_dim) {
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
        if (kv_init_shape[axis].is_static() && kv_init_shape[axis].get_length() == 0) {
            // this is a sequence dimension based on how the initialization expression is build for stateful models
            std::shared_ptr<ov::Node> replacement;
            if (prev_max_seq_len->get_output_element_type(0) != target_type) {
                replacement = std::make_shared<v0::Convert>(prev_max_seq_len, target_type);
            } else {
                replacement = prev_max_seq_len;
            }
            replace_node(
                gather,
                std::make_shared<v1::Reshape>(replacement, v0::Constant::create(element::i64, Shape{1}, {1}), false));
            return true;
        } else {  // assumption that any other axis should point to batch dimension, precise reasoning is too complex
                  // (TODO)
            // this is a batch dimension
            std::shared_ptr<ov::Node> replacement;
            if (batch_dim->get_output_element_type(0) != target_type) {
                replacement = std::make_shared<v0::Convert>(batch_dim, target_type);
            } else {
                replacement = batch_dim;
            }
            replace_node(gather, replacement);
            return true;
        }
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(seq, matcher_name);
    register_matcher(m, callback);
}
