// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/sdpa_to_paged_attention/total_sequence_length_pattern.hpp"

#include "openvino/cc/pass/itt.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::op;

ov::pass::TotalSequenceLengthPattern::TotalSequenceLengthPattern(
    const std::shared_ptr<ov::op::v0::Parameter>& max_context_len) {
    MATCHER_SCOPE(TotalSequenceLengthPattern);

    auto kv_past = pattern::wrap_type<v6::ReadValue>({pattern::any_input()});
    auto kv_gather = pattern::wrap_type<v8::Gather>({kv_past, pattern::any_input(), pattern::any_input()});
    auto kv_current = pattern::any_input();
    auto kv_concat = pattern::wrap_type<v0::Concat>({kv_gather, kv_current});
    auto kv_shape = pattern::wrap_type<v3::ShapeOf>({kv_concat});
    auto seq = pattern::wrap_type<v8::Gather>({kv_shape, pattern::any_input(), pattern::any_input()});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        // TODO: Check that seq has axis that really takes sequence len but not any other dimension --
        //  use symbolic infra or look at the constant input
        auto gather = m.get_match_root();
        auto target_type = gather->get_output_element_type(0);
        std::shared_ptr<Node> replacement = max_context_len;
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