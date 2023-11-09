// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stateful_sdp_fusion.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/common/op/sdp.hpp"

#define CALLBACK_LOG(m) std::cout << matcher_name << " " << m.get_match_root()->get_friendly_name() << std::endl;

namespace ov {
namespace intel_cpu {

StatefulSDPFusion::StatefulSDPFusion() {
    MATCHER_SCOPE(StatefulSDPFusion);
    using namespace ov::pass::pattern;

    auto past_k = wrap_type<opset6::ReadValue>();
    auto past_v = wrap_type<opset6::ReadValue>();
    auto concat_k = wrap_type<opset6::Concat>({past_k, any_input()});
    auto concat_v = wrap_type<opset6::Concat>({past_v, any_input()});
    auto sdp0 = wrap_type<opset13::ScaledDotProductAttention>({any_input(), concat_k, concat_v});
    auto sdp1 = wrap_type<opset13::ScaledDotProductAttention>({any_input(), concat_k, concat_v, any_input()});
    auto sdp2 = wrap_type<opset13::ScaledDotProductAttention>({any_input(), concat_k, concat_v, any_input(), any_input()});
    auto sdp = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sdp0, sdp1, sdp2});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        CALLBACK_LOG(m);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto find_assign = [&](const ov::Output<ov::Node>& out) -> opset6::Assign* {
            auto present_to = out.get_target_inputs();
            if (present_to.size() != 2)
                return nullptr;
            for (auto& to : present_to) {
                auto to_node = to.get_node();
                if (auto convert = dynamic_cast<opset1::Convert*>(to_node)) {
                    auto cvt_targets = convert->get_output_target_inputs(0);
                    if (cvt_targets.size() == 1) {
                        to_node = cvt_targets.begin()->get_node();
                    }
                }
                if (auto assign = dynamic_cast<opset6::Assign*>(to_node))
                    return assign;
            }
            return nullptr;
        };

        const auto sdp_node = ov::as_type_ptr<opset13::ScaledDotProductAttention>(root);
        const auto past_k_node = ov::as_type_ptr<opset6::ReadValue>(pattern_map.at(past_k).get_node_shared_ptr());
        const auto past_v_node = ov::as_type_ptr<opset6::ReadValue>(pattern_map.at(past_v).get_node_shared_ptr());
        const auto concat_k_node = ov::as_type_ptr<opset6::Concat>(pattern_map.at(concat_k).get_node_shared_ptr());
        const auto concat_v_node = ov::as_type_ptr<opset6::Concat>(pattern_map.at(concat_v).get_node_shared_ptr());
        auto* assign_k_node = find_assign(concat_k_node);
        if (!assign_k_node)
            return false;
        if (past_k_node->get_variable_id() != assign_k_node->get_variable_id())
            return false;

        auto* assign_v_node = find_assign(concat_v_node);
        if (!assign_v_node)
            return false;
        if (past_v_node->get_variable_id() != assign_v_node->get_variable_id())
            return false;

        auto args = sdp_node->input_values();
        args[1] = concat_k_node->input_value(1);
        args[2] = concat_v_node->input_value(1);
        args.push_back(past_k_node->output(0));
        args.push_back(past_v_node->output(0));
        ov::intel_cpu::ScaledDotProductAttentionNode::Config config;

        config.is_causal = sdp_node->get_causal();
        config.fuse_concat = true;

        auto old_node = sdp_node;
        auto new_node = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionNode>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, {new_node->output(0)});
        assign_k_node->set_arguments({new_node->output(1)});
        assign_v_node->set_arguments({new_node->output(2)});

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdp, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace intel_cpu
}  // namespace ov
