// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <iostream>

#include "sdpa.hpp"

#include "../../logging.hpp"
#include "../online/group.hpp"     // online::Group
#include "../online/snapshot.hpp"  // online::Snapshot
#include "openvino/op/ops.hpp"
#include "openvino/pass/pattern/op/label.hpp"  // any_input
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "openvino/util/common_util.hpp"

namespace ov {
namespace npuw {
namespace patterns {
namespace attn {

namespace opp = ov::pass::pattern;

SDPA::SDPA(const std::shared_ptr<ov::npuw::online::Snapshot>& snapshot, const std::string& isol_tag) {
    auto past_k_in = opp::wrap_type<ov::op::v0::Parameter>();
    auto past_k_cvt = opp::optional<ov::op::v0::Convert>({past_k_in->output(0)});
    auto past_k_cat = opp::wrap_type<ov::op::v0::Concat>({past_k_cvt, opp::any_input()});

    auto past_v_in = opp::wrap_type<ov::op::v0::Parameter>();
    auto past_v_cvt = opp::optional<ov::op::v0::Convert>({past_v_in->output(0)});
    auto past_v_cat = opp::wrap_type<ov::op::v0::Concat>({past_v_cvt, opp::any_input()});

    auto sdpa = opp::wrap_type<ov::op::v13::ScaledDotProductAttention>({opp::any_input(), past_k_cat, past_v_cat, opp::any_input(), opp::any_input()}
);
    auto trans = opp::wrap_type<ov::op::v1::Transpose>({sdpa, opp::any_input()});
    auto reshape = opp::wrap_type<ov::op::v1::Reshape>({trans, opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto pattern_nodes = std::vector<std::shared_ptr<ov::Node> > {
            past_k_in, past_k_cvt, past_k_cat, past_v_in, past_v_cvt, past_v_cat,
            sdpa, trans, reshape
        };
        for (auto &&pattern_node : pattern_nodes) {
            if (auto match_iter = node_to_output.find(pattern_node);
                match_iter != node_to_output.end()) {
                auto matched_node = match_iter->second.get_node_shared_ptr();
                if (auto group_iter = node_to_gptr->find(matched_node);
                    group_iter != node_to_gptr->end()) {
                    group_iter->second->isolate(isol_tag);
                }
            }
        }
        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(reshape, "TagSDPA"), std::move(callback));
}

}  // namespace attn
}  // namespace patterns
}  // namespace npuw
}  // namespace ov
