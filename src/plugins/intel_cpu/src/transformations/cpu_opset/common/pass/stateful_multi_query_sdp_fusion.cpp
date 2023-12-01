// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stateful_multi_query_sdp_fusion.hpp"

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

namespace ov {
namespace intel_cpu {

StatefulMultiQuerySDPFusion::StatefulMultiQuerySDPFusion() {
    MATCHER_SCOPE(StatefulMultiQuerySDPFusion);
    using namespace ov::pass::pattern;

    auto past_k = wrap_type<opset6::ReadValue>();
    auto past_v = wrap_type<opset6::ReadValue>();
    auto convert_past_k = wrap_type<opset1::Convert>({past_k});
    auto convert_past_v = wrap_type<opset1::Convert>({past_v});
    auto concat_input_k = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past_k, convert_past_k});
    auto concat_input_v = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{past_v, convert_past_v});
    auto concat_k = wrap_type<opset6::Concat>({concat_input_k, any_input()});
    auto concat_v = wrap_type<opset6::Concat>({concat_input_v, any_input()});
    auto reshape_k = wrap_type<opset6::Reshape>({concat_k, any_input()});
    auto reshape_v = wrap_type<opset6::Reshape>({concat_v, any_input()});
    auto constant_k = wrap_type<opset6::Constant>();
    auto constant_v = wrap_type<opset6::Constant>();
    auto multiply_k = wrap_type<opset6::Multiply>({reshape_k, constant_k});
    auto multiply_v = wrap_type<opset6::Multiply>({reshape_v, constant_v});
    auto reshape1_k = wrap_type<opset6::Reshape>({multiply_k, any_input()});
    auto reshape1_v = wrap_type<opset6::Reshape>({multiply_v, any_input()});
    auto order_k = wrap_type<opset6::Constant>();
    auto order_v = wrap_type<opset6::Constant>();
    auto transpose_k = wrap_type<opset6::Transpose>({reshape1_k, order_k});
    auto transpose_v = wrap_type<opset6::Transpose>({reshape1_v, order_v});

    auto order_q = wrap_type<opset6::Constant>();
    auto transpose_q = wrap_type<opset6::Transpose>({any_input(), order_q});
    auto sdp0 = wrap_type<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v});
    auto sdp1 = wrap_type<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v, any_input()});
    auto sdp2 = wrap_type<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v, any_input(), any_input()});
    auto sdp = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sdp0, sdp1, sdp2});

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        // const auto sdp_node = ov::as_type_ptr<opset13::ScaledDotProductAttention>(root);
        std::cout << "1Match Multi Query|" << root->get_friendly_name() << std::endl;
        auto find_assign = [&](const ov::Output<ov::Node>& out, opset6::Assign*& assign, opset1::Convert*& cvt) {
            auto present_to = out.get_target_inputs();
            if (present_to.size() != 2)
                return;
            for (auto& to : present_to) {
                auto to_node = to.get_node();
                if (auto convert = dynamic_cast<opset1::Convert*>(to_node)) {
                    auto cvt_targets = convert->get_output_target_inputs(0);
                    if (cvt_targets.size() == 1) {
                        to_node = cvt_targets.begin()->get_node();
                        cvt = convert;
                    }
                }
                assign = dynamic_cast<opset6::Assign*>(to_node);
                if (assign)
                    return;
            }
        };

        std::shared_ptr<opset1::Convert> read_cvt_k_node, read_cvt_v_node;
        const auto sdp_node = ov::as_type_ptr<opset13::ScaledDotProductAttention>(root);
        const auto past_k_node = ov::as_type_ptr<opset6::ReadValue>(pattern_map.at(past_k).get_node_shared_ptr());
        const auto past_v_node = ov::as_type_ptr<opset6::ReadValue>(pattern_map.at(past_v).get_node_shared_ptr());
        const auto concat_k_node = ov::as_type_ptr<opset6::Concat>(pattern_map.at(concat_k).get_node_shared_ptr());
        const auto concat_v_node = ov::as_type_ptr<opset6::Concat>(pattern_map.at(concat_v).get_node_shared_ptr());
        if (pattern_map.count(convert_past_k)) {
            read_cvt_k_node = ov::as_type_ptr<opset1::Convert>(pattern_map.at(convert_past_k).get_node_shared_ptr());
            read_cvt_v_node = ov::as_type_ptr<opset1::Convert>(pattern_map.at(convert_past_v).get_node_shared_ptr());
        }
        opset6::Assign* assign_k_node = nullptr, *assign_v_node = nullptr;
        opset1::Convert* assign_cvt_k_node = nullptr, *assign_cvt_v_node = nullptr;
        find_assign(concat_k_node, assign_k_node, assign_cvt_k_node);
        if (!assign_k_node)
            return false;
        if (past_k_node->get_variable_id() != assign_k_node->get_variable_id())
            return false;

        find_assign(concat_v_node, assign_v_node, assign_cvt_v_node);
        if (!assign_v_node)
            return false;
        if (past_v_node->get_variable_id() != assign_v_node->get_variable_id())
            return false;
        std::cout << "2Match Multi Query|Find Assign|" << root->get_friendly_name() << std::endl;
        auto args = sdp_node->input_values();
        std::cout << "args0 shape|" << pattern_map.at(transpose_q).get_node_shared_ptr()->input_value(0).get_partial_shape() << std::endl;
        args[0] = pattern_map.at(transpose_q).get_node_shared_ptr()->input_value(0);
        args[1] = concat_k_node->input_value(1);
        args[2] = concat_v_node->input_value(1);
        std::cout << "-2|" << past_k_node->output(0).get_partial_shape() << std::endl;
        args.push_back(read_cvt_k_node ? read_cvt_k_node->output(0) : past_k_node->output(0));
        args.push_back(read_cvt_v_node ? read_cvt_v_node->output(0) : past_v_node->output(0));
        ov::intel_cpu::ScaledDotProductAttentionStub::Config config;

        const auto order_k_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_k).get_node_shared_ptr());
        const auto order_v_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_v).get_node_shared_ptr());
        auto check_lbhs_input = [&](const std::shared_ptr<opset6::Constant>& node) {
            const std::vector<int32_t> order = node->get_vector<int32_t>();
            const std::vector<int32_t> target = {1, 2, 0, 3};
            if (order.size() != 4)
                return false;
            bool all_equal = true;
            for (size_t i = 0; i < target.size(); i++) {
                all_equal = all_equal && order[i] == target[i];
            }
            return all_equal;
        };

        if (!(check_lbhs_input(order_k_node) && check_lbhs_input(order_v_node)))
            return false;
        std::cout << "3Match Multi Query|check_lbhs_input|" << root->get_friendly_name() << std::endl;;
        config.is_causal = sdp_node->get_causal();
        config.fuse_concat = true;
        config.is_lbhs_input = true;
        config.is_multi_query = true;

        auto& old_node = sdp_node;
        auto new_node = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionStub>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, {new_node->output(0)});
        if (assign_cvt_k_node)
            assign_cvt_k_node->set_arguments({new_node->output(1)});
        else
            assign_k_node->set_arguments({new_node->output(1)});

        if (assign_cvt_v_node)
            assign_cvt_v_node->set_arguments({new_node->output(2)});
        else
            assign_v_node->set_arguments({new_node->output(2)});

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(sdp, matcher_name);
    this->register_matcher(m, callback);
}

}  // namespace intel_cpu
}  // namespace ov