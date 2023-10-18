// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "stateful_sdp_fusion.hpp"

#include <cstdint>
#include <limits>
#include <openvino/core/rt_info.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset12.hpp>
#include <openvino/opsets/opset6.hpp>
#include <openvino/opsets/opset8.hpp>
#include <openvino/pass/pattern/op/or.hpp>
#include <openvino/pass/pattern/op/wrap_type.hpp>
#include <transformations/utils/utils.hpp>

#include "itt.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/x64/op/rope.hpp"
#include "transformations/cpu_opset/x64/op/sdp.hpp"
#include "utils/gen_pattern.hpp"

#define CALLBACK_LOG(m) std::cout << matcher_name << " " << m.get_match_root()->get_friendly_name() << std::endl;

namespace ov {
namespace intel_cpu {

StatefulSDPFusion::StatefulSDPFusion() {
    MATCHER_SCOPE(StatefulSDPFusion);

    // Skip StatefulSDPFusion unless explicitly required
    if (!std::getenv("FUSE_SDP") || (atoi(std::getenv("FUSE_SDP")) == 0)) {
        return;
    }
    auto query_key_value_proj = GenPattern(ov::Rank(3));  // "f32[1,?,7680]"
    auto attn_mask = GenPattern(ov::Rank(4));
    auto position_ids = GenPattern(ov::Rank(4));  // "i32[?,1,?,20]"
    auto const_cos = GenPattern(ov::Rank(4));     // cos "f32[1,1,2048,20]"
    auto const_sin = GenPattern(ov::Rank(4));     // sin "f32[1,1,2048,20]"

    auto num_heads = Symbol("num_heads");
    auto num_states_per_head = Symbol("num_states_per_head");
    auto rope_ndims = Symbol("rope_ndims");

    auto ShapeOf_276155 = GenPattern<opset1::ShapeOf>({query_key_value_proj}, "i32[3]");
    auto Gather_285219 = GenPattern<opset8::Gather>({ShapeOf_276155, {0, 1}, {0}}, "i32[2]", {{"batch_dims", 0}});
    auto shape_4d =
        GenPattern<opset1::Concat>({Gather_285219, {num_heads}, {num_states_per_head * 3}}, "i32[4]", {{"axis", 0}});
    auto view_Reshape = GenPattern<opset1::Reshape>({query_key_value_proj, shape_4d},
                                                    nullptr,
                                                    {{"special_zero", 0}});  // "f32[1,?,32,240]"

    auto query = GenPattern<ov::intel_cpu::RoPENode>({view_Reshape, const_cos, const_sin, position_ids},
                                                     nullptr,
                                                     {{"config.slice_start", 0},
                                                      {"config.slice_stop", num_states_per_head},
                                                      {"config.input_trans0213", 1},
                                                      {"config.cos_is_raw3d", 0},
                                                      {"config.sin_is_raw3d", 0},
                                                      {"config.output_trans0213", 0},
                                                      {"config.ndims", rope_ndims},
                                                      {"config.gather_position_arg_id", 3},
                                                      {"config.concat_with_past_arg_id", 0}});  //  "f32[1,32,?,80]"
    auto ReadValue_past_key = GenPattern<opset6::ReadValue>({});  //  past_key_values.1.keypresent.1.key[1,32,?,80]f32/bf16
    auto ReadValue_past_key_cvt =
        GenPattern<opset1::Convert>({ReadValue_past_key}, nullptr, {{"destination_type", "f32"}});

    auto present_key = GenPattern<ov::intel_cpu::RoPENode>(
        {view_Reshape, const_cos, const_sin, position_ids, ReadValue_past_key | ReadValue_past_key_cvt},
        nullptr,
        {{"config.slice_start", num_states_per_head},
         {"config.slice_stop", num_states_per_head * 2},
         {"config.input_trans0213", 1},
         {"config.cos_is_raw3d", 0},
         {"config.sin_is_raw3d", 0},
         {"config.output_trans0213", 0},
         {"config.ndims", rope_ndims},
         {"config.gather_position_arg_id", 3},
         {"config.concat_with_past_arg_id", 4}});  //
    //auto Assign_past_key =
    //    GenPattern<opset6::Assign>({present_key});  //  past_key_values.1.keypresent.1.key[1,32,?,80]f32

    auto ReadValue_past_value =
        GenPattern<opset6::ReadValue>({}, nullptr);  // past_key_values.1.valuepresent.1.value[1,32,?,80]f32/bf16
    auto ReadValue_past_value_cvt =
        GenPattern<opset1::Convert>({ReadValue_past_value}, nullptr, {{"destination_type", "f32"}});

    auto slice_Slice_329 = GenPattern<opset1::StridedSlice>(
        {view_Reshape, {0, 0, 0, num_states_per_head * 2}, {0, 0, 0, 2147483647}, {1, 1, 1, 1}},
        nullptr,
        {{"begin_mask", {1, 1, 1, 0}},
         {"end_mask", {1, 1, 1, 0}},
         {"new_axis_mask", {}},
         {"shrink_axis_mask", {}},
         {"ellipsis_mask", {}}});  //  "f32[1,?,32,80]"
    auto permute_Transpose_330 = GenPattern<opset1::Transpose>({slice_Slice_329, {0, 2, 1, 3}}, nullptr);
    auto present_value =
        GenPattern<opset1::Concat>({ReadValue_past_value | ReadValue_past_value_cvt, permute_Transpose_330},
                                   nullptr,
                                   {{"axis", -2}});
    //auto Assign_past_value =
    //    GenPattern<opset6::Assign>({present_value});  // past_key_values.1.valuepresent.1.value[1,32,?,80]f32

    auto scaled_dot_product =
        GenPattern<opset12::ScaledDotProductAttention>({query, present_key, present_value, attn_mask},
                                                       nullptr,
                                                       {{"causal", 1}});

    auto permute_Transpose_500 = GenPattern<opset1::Transpose>({scaled_dot_product, {0, 2, 1, 3}});  // "f32[1,?,32,80]"
    auto ShapeOf_276636 = GenPattern<opset1::ShapeOf>({permute_Transpose_500}, "i32[4]");
    auto Gather_285224 = GenPattern<opset8::Gather>({ShapeOf_276636, {0, 1}, {0}}, "i32[2]", {{"batch_dims", 0}});
    auto ListConstruct_509_Concat =
        GenPattern<opset1::Concat>({Gather_285224, {num_heads * num_states_per_head}}, "i32[3]", {{"axis", 0}});
    auto view_Reshape_510 = GenPattern<opset1::Reshape>({permute_Transpose_500, ListConstruct_509_Concat},
                                                        nullptr,
                                                        {{"special_zero", 0}});  // "f32[1,?,2560]"

    matcher_pass_callback callback = [=](ngraph::pattern::Matcher& m) {
        CALLBACK_LOG(m);

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        std::map<std::string, double> symbol_name2value;
        if (!validate_matched_symbols(m, symbol_name2value)) {
            return false;
        }

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

        auto* present_key_assign = find_assign(pattern_map.at(present_key));
        if (!present_key_assign)
            return false;
        auto past_key_read = reinterpret_cast<opset6::ReadValue*>(pattern_map.at(ReadValue_past_key).get_node());
        if (past_key_read->get_variable_id() != present_key_assign->get_variable_id())
            return false;

        auto* present_value_assign = find_assign(pattern_map.at(present_value));
        if (!present_value_assign)
            return false;
        auto past_value_read = reinterpret_cast<opset6::ReadValue*>(pattern_map.at(ReadValue_past_value).get_node());
        if (past_value_read->get_variable_id() != present_value_assign->get_variable_id())
            return false;

        // markup for deletion
        present_key_assign->get_rt_info()["fused_into_sdp"] = true;
        present_value_assign->get_rt_info()["fused_into_sdp"] = true;

        OutputVector args{pattern_map.at(query_key_value_proj),
                          pattern_map.at(attn_mask),
                          pattern_map.at(const_cos),
                          pattern_map.at(const_sin),
                          pattern_map.at(position_ids)};
        ov::intel_cpu::ScaledDotProductAttentionNode::Config config;

        config.qkv_merged = true;
        config.input_trans0213 = true;
        config.cos_is_raw3d = false;
        config.sin_is_raw3d = false;
        config.output_BLHxS = true;
        config.rope_ndims = symbol_name2value["rope_ndims"];
        config.gather_position_arg_id = 4;
        config.m_is_causal = true;
        config.num_heads = symbol_name2value["num_heads"];
        config.num_states_per_head = symbol_name2value["num_states_per_head"];
        config.past_key_var = past_key_read->get_variable();
        config.past_value_var = past_value_read->get_variable();

        auto old_node = root;
        auto new_node = std::make_shared<ov::intel_cpu::ScaledDotProductAttentionNode>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::replace_node(old_node, new_node);

        // this new node may match following additional matchers
        // register_new_node(new_node);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(view_Reshape_510, matcher_name);
    this->register_matcher(m, callback);
}

bool RemoveFusedAssign::run_on_model(const std::shared_ptr<ov::Model>& m) {
    RUN_ON_MODEL_SCOPE(RemoveFusedAssign);

    bool is_changed = false;
    std::vector<std::shared_ptr<opset6::Assign>> assigns_to_remove;
    for (auto& sink : m->get_sinks()) {
        if (auto assign = std::dynamic_pointer_cast<opset6::Assign>(sink)) {
            auto& rt_info = assign->get_rt_info();
            if (rt_info.count("fused_into_sdp") && rt_info["fused_into_sdp"].as<bool>()) {
                assigns_to_remove.push_back(assign);
            }
        }
    }

    for (auto& assign : assigns_to_remove) {
        std::cout << " ========= remove assign to " << assign->get_variable_id() << std::endl;
        m->remove_sink(assign);
        is_changed = true;
    }
    return is_changed;
}

}  // namespace intel_cpu
}  // namespace ov
