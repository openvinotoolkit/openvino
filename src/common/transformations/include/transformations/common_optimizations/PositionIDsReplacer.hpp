// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/parameter.hpp"
#include "openvino/pass/graph_rewrite.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/cc/pass/itt.hpp"
#include "transformations_visibility.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace pass {

class TRANSFORMATIONS_API PositionIDsReplacer;

}  // namespace pass
}  // namespace ov

using namespace ov::op;

class ov::pass::PositionIDsReplacer : public ov::pass::MatcherPass {
public:
    OPENVINO_RTTI("PositionIDsReplacer", "0");

    // TODO: Instead of using the following transformation that matches quite a specific place in a model graph in case when position_ids parameter is missing,
    //        consider replacing always existing attention_mask parameter with a sub-graph using a new slot_mapping parameter.
    PositionIDsReplacer(const std::shared_ptr<Output<Node>>& position_ids) {
        MATCHER_SCOPE(PositionIDsReplacer);

        auto input_ids = pattern::any_input();
        auto input_embed = pattern::wrap_type<v1::Gather>({pattern::any_input(), input_ids, pattern::any_input()});

        auto position_ids_pattern = pattern::any_input();
        auto offset = pattern::wrap_type<v0::Constant>();
        auto add_offset = pattern::wrap_type<v1::Add>({position_ids_pattern, offset});
        auto convert = pattern::wrap_type<v0::Convert>({add_offset});
        auto position_embed = pattern::wrap_type<v1::Gather>({pattern::any_input(), convert, pattern::any_input()});

        auto add = pattern::wrap_type<v1::Add>({input_embed, position_embed});

        ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
            const auto& pattern_map = m.get_pattern_map();
            replace_node(pattern_map.at(position_ids_pattern), position_ids->get_node_shared_ptr());
            std::cout << "APPLIED position_ids PARAMETER INSTEAD OF attention_mask-BASED SUB-GRAPH" << std::endl;
            return true;
        };

        auto m = std::make_shared<ov::pass::pattern::Matcher>(add, matcher_name);
        register_matcher(m, callback);
    }
};