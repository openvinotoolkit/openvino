// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stateful_sdpa_fusion.hpp"

#include <utils/general_utils.h>

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include "openvino/opsets/opset1.hpp"
#include <openvino/opsets/opset13.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/common/op/sdpa.hpp"
#include "utils/gen_pattern.hpp"
using namespace ov::gen_pattern;

namespace ov {
namespace intel_cpu {

StatefulSDPAFusion::StatefulSDPAFusion() {
    MATCHER_SCOPE(StatefulSDPAFusion);
    using namespace ov::pass::pattern;

    auto beam_idx = makePattern("i32[?]");
    auto cur_q = any_input();
    auto cur_k = any_input();
    auto cur_v = any_input();

    auto axis_seq_len = Symbol("axis_seq_len");
    auto axis_beam = Symbol("axis_beam");

    // past_kv can be BHLS/LBHS
    auto past_k = makePattern<opset6::ReadValue>({});
    auto past_v = makePattern<opset6::ReadValue>({});

    auto convert_past_k = makePattern<opset1::Convert>({past_k});
    auto convert_past_v = makePattern<opset1::Convert>({past_v});

    auto gather_input_k =
        makePattern<opset8::Gather>({past_k | convert_past_k, beam_idx, axis_beam}, {{"batch_dims", 0}});
    auto gather_input_v =
        makePattern<opset8::Gather>({past_v | convert_past_v, beam_idx, axis_beam}, {{"batch_dims", 0}});

    auto concat_k = makePattern<opset1::Concat>({gather_input_k, cur_k}, {{"axis", axis_seq_len}});
    auto concat_v = makePattern<opset1::Concat>({gather_input_v, cur_v}, {{"axis", axis_seq_len}});

    auto multi_query_bcst = [](std::shared_ptr<Node> kv) {
        auto reshape_kv = wrap_type<opset6::Reshape>({kv, any_input()});
        auto unsqueeze_kv = makePattern<opset1::Unsqueeze>({kv, -2});
        auto constant_bcst = makeConst(ov::element::f32, ov::PartialShape("[...]"), [](ov::op::v0::Constant& node) {
            const auto& bcst_arg = node.cast_vector<float>();
            return std::all_of(bcst_arg.begin(), bcst_arg.end(), [](float i) {
                return i == 1.0;
            });
        });
        auto multiply_kv = wrap_type<opset6::Multiply>({reshape_kv | unsqueeze_kv, constant_bcst});
        return wrap_type<opset6::Reshape>({multiply_kv, any_input()});
    };

    auto present_k = concat_k | multi_query_bcst(concat_k);
    auto present_v = concat_v | multi_query_bcst(concat_v);

    // canonical q/k/v shape definition: [B,H,...L,S]
    auto sdp0 = makePattern<opset13::ScaledDotProductAttention>({cur_q, present_k, present_v});
    auto sdp1 = makePattern<opset13::ScaledDotProductAttention>({cur_q, present_k, present_v, any_input()});
    auto sdp2 =
        makePattern<opset13::ScaledDotProductAttention>({cur_q, present_k, present_v, any_input(), any_input()});

    // non-canonical q/k/v shape definitions, for example: [L, B, H, S]/[B, L, H, S]
    auto order_k = wrap_type<opset6::Constant>();
    auto order_v = wrap_type<opset6::Constant>();
    auto order_q = wrap_type<opset6::Constant>();
    auto transpose_q = makePattern<opset6::Transpose>({cur_q, order_q});
    auto transpose_k = makePattern<opset1::Transpose>({present_k, order_k});
    auto transpose_v = makePattern<opset1::Transpose>({present_v, order_v});

    auto sdp_trans0 = makePattern<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v});
    auto sdp_trans1 =
        makePattern<opset13::ScaledDotProductAttention>({transpose_q, transpose_k, transpose_v, any_input()});
    auto sdp_trans2 = makePattern<opset13::ScaledDotProductAttention>(
        {transpose_q, transpose_k, transpose_v, any_input(), any_input()});

    auto sdp = sdp0 | sdp1 | sdp2 | sdp_trans0 | sdp_trans1 | sdp_trans2;

    ov::matcher_pass_callback callback = [=](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        auto find_assign = [&](const ov::Output<ov::Node>& out, opset6::Assign*& assign, opset1::Convert*& cvt) {
            auto present_to = out.get_target_inputs();
            if (present_to.size() != 2)
                return false;
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
                    return true;
            }
            return false;
        };
        auto check_valid_children_type = [](const ov::Output<ov::Node>& out) {
            auto children = out.get_target_inputs();
            for (auto& child : children) {
                auto node = child.get_node();
                if (!one_of(node->get_type_info(),
                            ov::op::v13::ScaledDotProductAttention::get_type_info_static(),
                            ov::op::v0::ShapeOf::get_type_info_static(),
                            ov::op::v3::ShapeOf::get_type_info_static(),
                            ov::op::v0::Convert::get_type_info_static(),
                            ov::op::v8::Gather::get_type_info_static()))
                    return false;
            }
            return true;
        };

        const auto sdp_node = ov::as_type_ptr<opset13::ScaledDotProductAttention>(root);
        const auto past_k_node = ov::as_type_ptr<opset6::ReadValue>(pattern_map.at(past_k).get_node_shared_ptr());
        const auto past_v_node = ov::as_type_ptr<opset6::ReadValue>(pattern_map.at(past_v).get_node_shared_ptr());
        if (!check_valid_children_type(past_k_node) || !check_valid_children_type(past_v_node)) {
            return false;
        }
        const auto concat_k_node = ov::as_type_ptr<opset6::Concat>(pattern_map.at(concat_k).get_node_shared_ptr());
        const auto concat_v_node = ov::as_type_ptr<opset6::Concat>(pattern_map.at(concat_v).get_node_shared_ptr());

        opset6::Assign *assign_k_node = nullptr, *assign_v_node = nullptr;
        opset1::Convert *assign_cvt_k_node = nullptr, *assign_cvt_v_node = nullptr;
        if (!find_assign(concat_k_node, assign_k_node, assign_cvt_k_node))
            return false;
        if (past_k_node->get_variable_id() != assign_k_node->get_variable_id())
            return false;

        if (!find_assign(concat_v_node, assign_v_node, assign_cvt_v_node))
            return false;
        if (past_v_node->get_variable_id() != assign_v_node->get_variable_id())
            return false;

        // past_k & past_v must be reordered by same beam_idx
        const auto gather_k_node =
            ov::as_type_ptr<opset8::Gather>(pattern_map.at(gather_input_k).get_node_shared_ptr());
        const auto gather_v_node =
            ov::as_type_ptr<opset8::Gather>(pattern_map.at(gather_input_v).get_node_shared_ptr());
        if (gather_k_node->input_value(1) != gather_v_node->input_value(1)) {
            return false;
        }

        OutputVector args = sdp_node->input_values();
        args[0] = pattern_map.at(cur_q);
        args[1] = pattern_map.at(cur_k);
        args[2] = pattern_map.at(cur_v);
        args.push_back(pattern_map.at(beam_idx));
        args.push_back(gather_k_node->input_value(0));
        args.push_back(gather_v_node->input_value(0));
        ov::intel_cpu::ScaledDotProductAttentionWithKVCache::Config config;

        config.is_causal = sdp_node->get_causal();
        config.fuse_concat = true;

        if (pattern_map.count(order_q) && pattern_map.count(order_k) && pattern_map.count(order_v)) {
            const auto order_q_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_q).get_node_shared_ptr());
            const auto order_k_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_k).get_node_shared_ptr());
            const auto order_v_node = ov::as_type_ptr<opset6::Constant>(pattern_map.at(order_v).get_node_shared_ptr());
            const auto& permute_q = order_q_node->cast_vector<int32_t>();
            const auto& permute_k = order_k_node->cast_vector<int32_t>();
            const auto& permute_v = order_v_node->cast_vector<int32_t>();
            if (permute_q != permute_k || permute_q != permute_v) {
                return false;
            }
            config.permute_axes.resize(permute_q.size());
            for (size_t i = 0; i < permute_q.size(); i++) {
                config.permute_axes[i] = static_cast<size_t>(permute_q[i]);
            }
        }

        auto old_node = sdp_node;
        auto new_node = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionWithKVCache>(args, config);
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
