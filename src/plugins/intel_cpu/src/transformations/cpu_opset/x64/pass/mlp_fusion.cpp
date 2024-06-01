// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "mlp_fusion.hpp"

#include <cstdint>
#include <iostream>
#include <limits>

#include "itt.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/opsets/opset6.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/x64/op/llm_mlp.hpp"
#include "transformations/utils/utils.hpp"
#include "utils/gen_pattern.hpp"

using namespace ov::gen_pattern;

ov::intel_cpu::MLPFusion::MLPFusion() {
    MATCHER_SCOPE(MLPFusion);

    auto input = makePattern("f32[?,?,4096]");

    auto gate_proj_weight_compressed = makePattern<opset1::Constant>({});  // [11008,4096]
    auto gate_proj_weight =
        makePattern<opset1::Convert>({gate_proj_weight_compressed},
                                     {{"destination_type", "f32"}});     //  tensor_array<f32[11008,4096]>
    auto up_proj_weight_compressed = makePattern<opset1::Constant>({});  // [11008,4096]
    auto up_proj_weight =
        makePattern<opset1::Convert>({up_proj_weight_compressed},
                                     {{"destination_type", "f32"}});       //  tensor_array<f32[11008,4096]>
    auto down_proj_weight_compressed = makePattern<opset1::Constant>({});  // 4096,11008
    auto down_proj_weight =
        makePattern<opset1::Convert>({down_proj_weight_compressed},
                                     {{"destination_type", "f32"}});  //  tensor_array<f32[4096,11008]>
    auto mlp_gate_proj =
        makePattern<opset1::MatMul>({input, gate_proj_weight | gate_proj_weight_compressed},
                                    {{"transpose_a", false}, {"transpose_b", true}});  //  tensor_array<f32[?,?,11008]>
    auto mlp_silu_gate = makePattern<opset4::Swish>({mlp_gate_proj});                  //  tensor_array<f32[?,?,11008]>
    auto mlp_up_proj =
        makePattern<opset1::MatMul>({input, up_proj_weight | up_proj_weight_compressed},
                                    {{"transpose_a", false}, {"transpose_b", true}});  //  tensor_array<f32[?,?,11008]>
    auto mlp_gated_up = makePattern<opset1::Multiply>({mlp_silu_gate, mlp_up_proj},
                                                      {{"auto_broadcast", "numpy"}});  //  tensor_array<f32[?,?,11008]>
    auto down_proj =
        makePattern<opset1::MatMul>({mlp_gated_up, down_proj_weight | down_proj_weight_compressed},
                                    {{"transpose_a", false}, {"transpose_b", true}});  //  tensor_array<f32[?,?,4096]>

    auto result = down_proj;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        if (std::getenv("NOMLP")) {
            return false;
        }

        LLMMLPNode::Config config;
        OutputVector new_args;
        config.is_qkv_proj = false;
        config.hidden_size = 4096;
        config.intermediate_size = 11008;

        new_args.push_back(pattern_map.at(input));
        new_args.push_back(pattern_map.at(gate_proj_weight_compressed));
        new_args.push_back(pattern_map.at(up_proj_weight_compressed));
        new_args.push_back(pattern_map.at(down_proj_weight_compressed));

        auto old_node = root;
        auto new_node = std::make_shared<LLMMLPNode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(mlp_gate_proj).get_node_shared_ptr(),
                               pattern_map.at(mlp_silu_gate).get_node_shared_ptr(),
                               pattern_map.at(mlp_up_proj).get_node_shared_ptr(),
                               pattern_map.at(down_proj).get_node_shared_ptr()},
                              new_node);

        ov::replace_node(old_node, new_node);

        // this new node may match following additional matchers
        // register_new_node(new_node);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::QKVProjFusion::QKVProjFusion() {
    MATCHER_SCOPE(QKVProjFusion);

    auto input = makePattern("f32[?,?,4096]");

    // auto q_proj_weight_compressed = makeConst(element::f16, ov::Shape({4096,4096,}), {...});
    auto q_proj_weight = makePattern<opset1::Constant>({});
    auto q_proj_weight_avt = makePattern<opset1::Convert>({q_proj_weight}, {{"destination_type", "f32"}});   //  tensor_array<f32[4096,4096]>
    auto q_proj = makePattern<opset1::MatMul>({input, q_proj_weight_avt | q_proj_weight}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,?,4096]>

    /*
    auto k_proj_weight_compressed = makeConst(element::f16, ov::Shape({4096,4096,}), {...});
    auto k_proj_weight = makePattern<opset1::Convert>({k_proj_weight_compressed}, {{"destination_type", "f32"}});   //  tensor_array<f32[4096,4096]>
    auto k_proj = makePattern<opset1::MatMul>({input, k_proj_weight}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,?,4096]>

    auto v_proj_weight_compressed = makeConst(element::f16, ov::Shape({4096,4096,}), {...});
    auto v_proj_weight = makePattern<opset1::Convert>({v_proj_weight_compressed}, {{"destination_type", "f32"}});   //  tensor_array<f32[4096,4096]>
    auto v_proj = makePattern<opset1::MatMul>({input, v_proj_weight}, {{"transpose_a", false}, {"transpose_b", true}});   //  tensor_array<f32[?,?,4096]>
    */
    auto result = q_proj;

    matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto src = pattern_map.at(input);

        auto&& children = src.get_target_inputs();

        if (children.size() < 3)
            return false;

        OutputVector args = {src};
        OutputVector outputs;
        for (auto& child : children) {
            auto mm = dynamic_cast<opset1::MatMul*>(child.get_node());
            if (!mm) {
                continue;
            }
            auto constw = ov::as_type_ptr<opset1::Constant>(mm->input_value(1).get_node_shared_ptr());
            if (!constw) {
                auto cvt = ov::as_type_ptr<opset1::Convert>(mm->input_value(1).get_node_shared_ptr());
                if (!cvt) {
                    return false;
                }
                constw = ov::as_type_ptr<opset1::Constant>(cvt->input_value(0).get_node_shared_ptr());
            }
            if (!constw)
                return false;
            args.push_back(constw);
            outputs.push_back(mm->get_default_output());
        }

        if (std::getenv("NOQKV")) {
            return false;
        }

        //for(auto& arg: args)
        //    std::cout << "\t arg: " << arg << std::endl;
        //for(auto& out: outputs)
        //    std::cout << "\t out: " << out << std::endl;

        LLMMLPNode::Config config;
        config.is_qkv_proj = true;
        config.hidden_size = 4096;
        config.intermediate_size = 4096*3;

        auto old_node = root;
        auto new_node = std::make_shared<LLMMLPNode>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({old_node}, new_node);

        for (size_t i = 0; i < outputs.size(); i++) {
            //ov::replace_node(outputs[i].get_node_shared_ptr(), new_node, {int64_t(i)});

            auto target = outputs[i].get_node_shared_ptr();
            outputs[i].replace(new_node->output(i));
            new_node->add_node_control_dependents(target);
            new_node->add_node_control_dependencies(target);
            target->clear_control_dependents();
        }
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}
