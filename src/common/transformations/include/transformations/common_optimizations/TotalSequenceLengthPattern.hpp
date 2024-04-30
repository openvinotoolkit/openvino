// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API TotalSequenceLengthPattern;

}  // namespace pass
}  // namespace ov

using namespace ov::op;

class ov::pass::TotalSequenceLengthPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("TotalSequenceLengthPattern", "0");
    TotalSequenceLengthPattern(const std::shared_ptr<ov::op::v0::Parameter>& max_context_len) {
        MATCHER_SCOPE(TotalSequenceLengthPattern);

        auto kv_past = pattern::wrap_type<v6::ReadValue>({pattern::any_input()});
        auto kv_gather = pattern::wrap_type<v8::Gather>({kv_past, pattern::any_input(), pattern::any_input()});
        auto kv_current = pattern::any_input();
        auto kv_concat = pattern::wrap_type<v0::Concat>({kv_gather, kv_current});
        auto kv_shape = pattern::wrap_type<v3::ShapeOf>({kv_concat});
        auto seq = pattern::wrap_type<v8::Gather>({kv_shape, pattern::any_input(), pattern::any_input()});

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            // TODO: Check that seq has axis that really takes sequence len but not any other dimension -- use symbolic infra or look at the constant input
            std::cout << "____" << matcher_name << "___Matched___" << std::endl;
            auto gather = m.get_match_root();
            auto target_type = gather->get_output_element_type(0);
            std::shared_ptr<Node> replacement;
            if (max_context_len->get_output_element_type(0) != target_type) {
                std::cout << "Converting " << max_context_len->get_output_element_type(0) << " of total_seq_len to " << target_type << std::endl;
                replacement = std::make_shared<v0::Convert>(max_context_len, target_type);
            } else {
                replacement = max_context_len;
            }
            replace_node(gather, replacement);
            std::cout << "DETECTED PATTERN TotalSequenceLengthPattern, CONNECTED TO A DEDICATED PARAMETER" << std::endl;
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(seq, matcher_name);
        register_matcher(m, callback);
    }
};