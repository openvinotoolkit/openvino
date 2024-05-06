// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sdpa_to_paged_attention/prev_sequence_length_pattern.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

ov::pass::PrevSequenceLengthPattern::PrevSequenceLengthPattern(
    const std::shared_ptr<ov::op::v1::Subtract>& prev_max_seq_len) {
    MATCHER_SCOPE(PrevSequenceLengthPattern);

    auto kv_past = pattern::wrap_type<v6::ReadValue>({pattern::any_input()});
    auto kv_gather = pattern::wrap_type<v8::Gather>({kv_past, pattern::any_input(), pattern::any_input()});
    auto kv_shape = pattern::wrap_type<v3::ShapeOf>({kv_gather});
    auto seq = pattern::wrap_type<v8::Gather>({kv_past, pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        // TODO: Check that seq has axis that really takes sequence len but not any other dimension -- use symbolics or
        // look at the constant input
        auto gather = m.get_match_root();
        auto target_type = gather->get_output_element_type(0);
        std::shared_ptr<Node> replacement;
        if (prev_max_seq_len->get_output_element_type(0) != target_type) {
            replacement = std::make_shared<v0::Convert>(prev_max_seq_len, target_type);
        } else {
            replacement = prev_max_seq_len;
        }
        replace_node(gather, replacement);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(seq, matcher_name);
    register_matcher(m, callback);
}
