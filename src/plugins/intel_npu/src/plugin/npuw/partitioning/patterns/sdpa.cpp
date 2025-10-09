// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sdpa.hpp"

#include <iostream>

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

    // Optional part, probably one of many. Replace by graph traversal!
    auto opt_unsq_k = opp::optional<ov::op::v0::Unsqueeze>({past_k_cat->output(0), opp::any_input()});
    auto opt_bcast_k = opp::optional<ov::op::v3::Broadcast>({opt_unsq_k->output(0), opp::any_input()});
    auto opt_rshp_k = opp::optional<ov::op::v1::Reshape>({opt_bcast_k->output(0), opp::any_input()});

    auto opt_unsq_v = opp::optional<ov::op::v0::Unsqueeze>({past_v_cat->output(0), opp::any_input()});
    auto opt_bcast_v = opp::optional<ov::op::v3::Broadcast>({opt_unsq_v->output(0), opp::any_input()});
    auto opt_rshp_v = opp::optional<ov::op::v1::Reshape>({opt_bcast_v->output(0), opp::any_input()});

    auto sdpa = opp::wrap_type<ov::op::v13::ScaledDotProductAttention>(
        {opp::any_input(), opt_rshp_k, opt_rshp_v, opp::any_input(), opp::any_input()});
    auto trans = opp::wrap_type<ov::op::v1::Transpose>({sdpa, opp::any_input()});
    auto reshape = opp::wrap_type<ov::op::v1::Reshape>({trans, opp::any_input()});

    auto node_to_gptr = snapshot->getNodeToGroupMap();

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto pattern_nodes = std::vector<std::shared_ptr<ov::Node>>{past_k_in,
                                                                    past_k_cvt,
                                                                    past_k_cat,
                                                                    past_v_in,
                                                                    past_v_cvt,
                                                                    past_v_cat,
                                                                    opt_unsq_k,
                                                                    opt_bcast_k,
                                                                    opt_rshp_k,
                                                                    opt_unsq_v,
                                                                    opt_bcast_v,
                                                                    opt_rshp_v,
                                                                    sdpa,
                                                                    trans,
                                                                    reshape};
        for (auto&& pattern_node : pattern_nodes) {
            if (auto match_iter = node_to_output.find(pattern_node); match_iter != node_to_output.end()) {
                auto matched_node = match_iter->second.get_node_shared_ptr();
                if (auto group_iter = node_to_gptr->find(matched_node); group_iter != node_to_gptr->end()) {
                    group_iter->second->isolate(isol_tag);
                }
            }
        }
        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(reshape, "TagSDPA"), std::move(callback));
}

}  // namespace attn

namespace regularize {

namespace opp = ov::pass::pattern;

AttentionBroadcast::AttentionBroadcast() {
    // NB(dm): We've seen cases where this dynamic subgraph is placed on the K-path,
    // but I'd expect it could be on the V-path as well - so _kv in the name
    auto past_kv_in = opp::wrap_type<ov::op::v0::Parameter>();
    auto past_kv_cvt = opp::optional<ov::op::v0::Convert>({past_kv_in->output(0)});
    auto past_kv_cat = opp::wrap_type<ov::op::v0::Concat>({past_kv_cvt, opp::any_input()});

    // The dynamic shape calculation to be eliminated
    // NB: It only works in static shape graphs
    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({past_kv_cat});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    // NB: THREE inputs is also a case (see below)
    auto concat = opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input(), opp::any_input()});

    // Broadcast - the consumer
    auto unsq_kv = opp::wrap_type<ov::op::v0::Unsqueeze>({past_kv_cat, opp::any_input()});
    auto bcast_kv = opp::wrap_type<ov::op::v3::Broadcast>({unsq_kv, concat});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_concat_out = node_to_output.at(concat);
        auto& matched_concat_tensor = matched_concat_out.get_tensor();
        if (matched_concat_tensor.has_and_set_bound()) {
            // Replace the dynamic shape calculation with a static constant
            // This is bad but it in the current realm it is what it is
            auto new_const = std::make_shared<ov::op::v0::Constant>(matched_concat_tensor.get_upper_value());
            new_const->set_friendly_name("NPUW/Precalculated/" +
                                         matched_concat_out.get_node_shared_ptr()->get_friendly_name());
            for (auto&& input : matched_concat_out.get_target_inputs()) {
                input.replace_source_output(new_const);
            }
        }
        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(bcast_kv, "AttentionBroadcast"), std::move(callback));
}

// FIXME: Same as above but Concat has three inputs instead of four
AttentionBroadcast2::AttentionBroadcast2() {
    auto past_kv_in = opp::wrap_type<ov::op::v0::Parameter>();
    auto past_kv_cvt = opp::optional<ov::op::v0::Convert>({past_kv_in->output(0)});
    auto past_kv_cat = opp::wrap_type<ov::op::v0::Concat>({past_kv_cvt, opp::any_input()});

    auto shape_of = opp::wrap_type<ov::op::v3::ShapeOf>({past_kv_cat});
    auto gather = opp::wrap_type<ov::op::v8::Gather>({shape_of, opp::any_input(), opp::any_input()});
    auto concat =
        opp::wrap_type<ov::op::v0::Concat>({gather, opp::any_input(), opp::any_input()});  // THIS IS the difference

    // FIXME: using past_kv_cat as a 0th argument to this Unsqueeze breaks the pattern
    // for Phi-4
    auto unsq_kv = opp::wrap_type<ov::op::v0::Unsqueeze>({opp::any_input(), opp::any_input()});
    auto bcast_kv = opp::wrap_type<ov::op::v3::Broadcast>({unsq_kv, concat});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        auto matched_concat_out = node_to_output.at(concat);
        auto& matched_concat_tensor = matched_concat_out.get_tensor();
        if (matched_concat_tensor.has_and_set_bound()) {
            auto new_const = std::make_shared<ov::op::v0::Constant>(matched_concat_tensor.get_upper_value());
            new_const->set_friendly_name("NPUW/Precalculated/" +
                                         matched_concat_out.get_node_shared_ptr()->get_friendly_name());
            for (auto&& input : matched_concat_out.get_target_inputs()) {
                input.replace_source_output(new_const);
            }
        }
        return false;  // root hasn't changed
    };
    register_matcher(std::make_shared<opp::Matcher>(bcast_kv, "AttentionBroadcast2"), std::move(callback));
}

ShapeOfParameter::ShapeOfParameter() {
    auto param_in = opp::wrap_type<ov::op::v0::Parameter>();
    auto param_cvt = opp::wrap_type<ov::op::v0::Convert>({param_in});
    auto param_shp = opp::wrap_type<ov::op::v3::ShapeOf>({param_cvt});

    // Note: Use [=] to make sure the above objects stay alive in the callback
    auto callback = [=](ov::pass::pattern::Matcher& m) {
        auto& node_to_output = m.get_pattern_value_map();
        // Replace the path with a known constant too
        auto matched_shape_out = node_to_output.at(param_shp);
        auto& matched_shape_tensor = matched_shape_out.get_tensor();
        if (matched_shape_tensor.has_and_set_bound()) {
            auto new_const = std::make_shared<ov::op::v0::Constant>(matched_shape_tensor.get_upper_value());
            new_const->set_friendly_name("NPUW/Precalculated/" +
                                         matched_shape_out.get_node_shared_ptr()->get_friendly_name());
            for (auto&& input : matched_shape_out.get_target_inputs()) {
                input.replace_source_output(new_const);
            }
        }
        return false;  // root hasn't changed (?)
    };
    register_matcher(std::make_shared<opp::Matcher>(param_shp, "ShapeOfParameter"), std::move(callback));
}

}  // namespace regularize

}  // namespace patterns
}  // namespace npuw
}  // namespace ov
