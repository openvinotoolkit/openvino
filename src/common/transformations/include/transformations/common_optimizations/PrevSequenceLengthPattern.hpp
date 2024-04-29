// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations_visibility.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/shape_of.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PrevSequenceLengthPattern;

}  // namespace pass
}  // namespace ov

using namespace ov::op;

class ov::pass::PrevSequenceLengthPattern : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PrevSequenceLengthPattern", "0");
    PrevSequenceLengthPattern(const std::shared_ptr<ov::op::v1::Subtract>& prev_max_seq_len) {
        MATCHER_SCOPE(PrevSequenceLengthPattern);
 
        auto kv_past = pattern::wrap_type<v6::ReadValue>({pattern::any_input()});
        auto kv_gather = pattern::wrap_type<v8::Gather>({kv_past, pattern::any_input(), pattern::any_input()});
        auto kv_shape = pattern::wrap_type<v0::ShapeOf>({kv_gather});
        auto seq = pattern::wrap_type<v8::Gather>({kv_past, pattern::any_input(), pattern::any_input()});
 
        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        std::cout << "____" << matcher_name << "___Matched___" << std::endl;
            // TODO: Check that seq has axis that really takes sequence len but not any other dimension -- use symbolics or look at the constant input
            auto gather = m.get_match_root();
            auto target_type = gather->get_output_element_type(0);
            std::shared_ptr<Node> replacement;
            if (prev_max_seq_len->get_output_element_type(0) != target_type) {
                // std::cout << "Converting " << prev_max_seq_len->get_output_element_type(0) << " of max_context_len to " << target_type << std::endl;
                replacement = std::make_shared<v0::Convert>(prev_max_seq_len, target_type);
            } else {
                replacement = prev_max_seq_len;
            }
            replace_node(gather, replacement);
            // std::cout << "DETECTED PATTERN PrevSequenceLengthPattern, CONNECTED TO A DEDICATED PARAMETER" << std::endl;
            return true;
        };
 
        auto m = std::make_shared<ov::pass::pattern::Matcher>(seq, matcher_name);
        register_matcher(m, callback);
    }
};