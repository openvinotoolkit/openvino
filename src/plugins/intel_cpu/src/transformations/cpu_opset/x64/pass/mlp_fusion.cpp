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

    auto input = makePattern("f32[?,?,?]");

    auto gate_proj_weight_compressed = makePattern<opset1::Constant>({});  // [up_size, down_size]
    auto gate_proj_weight = makePattern<opset1::Convert>({gate_proj_weight_compressed}, {{"destination_type", "f32"}});
    auto up_proj_weight_compressed = makePattern<opset1::Constant>({});  // [up_size, down_size]
    auto up_proj_weight = makePattern<opset1::Convert>({up_proj_weight_compressed}, {{"destination_type", "f32"}});
    auto down_proj_weight_compressed = makePattern<opset1::Constant>({});  // [down_size, up_size]
    auto down_proj_weight = makePattern<opset1::Convert>({down_proj_weight_compressed}, {{"destination_type", "f32"}});
    auto mlp_gate_proj = makePattern<opset1::MatMul>({input, gate_proj_weight | gate_proj_weight_compressed},
                                                     {{"transpose_a", false}, {"transpose_b", true}});  // [?,?,up_size]
    auto mlp_silu_gate = makePattern<opset4::Swish>({mlp_gate_proj});
    auto mlp_up_proj = makePattern<opset1::MatMul>({input, up_proj_weight | up_proj_weight_compressed},
                                                   {{"transpose_a", false}, {"transpose_b", true}});
    auto mlp_gated_up = makePattern<opset1::Multiply>({mlp_silu_gate, mlp_up_proj}, {{"auto_broadcast", "numpy"}});
    auto down_proj = makePattern<opset1::MatMul>({mlp_gated_up, down_proj_weight | down_proj_weight_compressed},
                                                 {{"transpose_a", false}, {"transpose_b", true}});  //  [?,?,down_size]

    auto result = down_proj;

    matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto gate_proj_w = pattern_map.at(gate_proj_weight_compressed);
        auto up_proj_w = pattern_map.at(up_proj_weight_compressed);
        auto down_proj_w = pattern_map.at(down_proj_weight_compressed);

        auto gate_proj_w_pshape = gate_proj_w.get_partial_shape();
        auto up_proj_w_pshape = up_proj_w.get_partial_shape();
        auto down_proj_w_pshape = down_proj_w.get_partial_shape();

        // make sure that:
        //  - shape of gate/up's weight is [down_size, up_size]
        //  - shape of down's weight is [up_size, down_size]
        if (!gate_proj_w_pshape.is_static())
            return false;
        if (!up_proj_w_pshape.is_static())
            return false;
        if (!down_proj_w_pshape.is_static())
            return false;

        auto up_shape = up_proj_w_pshape.get_shape();
        auto down_shape = down_proj_w_pshape.get_shape();

        if (gate_proj_w_pshape.get_shape() != up_shape)
            return false;
        if (up_shape.size() != 2)
            return false;
        if (down_shape.size() != 2)
            return false;

        auto up_size = up_shape[0];
        auto down_size = up_shape[1];
        if (down_shape[0] != down_size)
            return false;
        if (down_shape[1] != up_size)
            return false;

        LLMMLPNode::Config config;
        OutputVector new_args;
        config.is_qkv_proj = false;
        config.hidden_size = down_size;
        config.intermediate_size = up_size;

        new_args.push_back(pattern_map.at(input));
        new_args.push_back(gate_proj_w);
        new_args.push_back(up_proj_w);
        new_args.push_back(down_proj_w);

        auto old_node = root;
        auto new_node = std::make_shared<LLMMLPNode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(mlp_gate_proj).get_node_shared_ptr(),
                               pattern_map.at(mlp_silu_gate).get_node_shared_ptr(),
                               pattern_map.at(mlp_up_proj).get_node_shared_ptr(),
                               pattern_map.at(down_proj).get_node_shared_ptr()},
                              new_node);

        ov::replace_node(old_node, new_node);

        std::cout << "MLPFusion:" << old_node->get_friendly_name() << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}

ov::intel_cpu::QKVProjFusion::QKVProjFusion() {
    MATCHER_SCOPE(QKVProjFusion);

    auto input = makePattern("f32[?,?,?]");

    auto q_proj_weight = makePattern<opset1::Constant>({});
    auto q_proj_weight_cvt =
        makePattern<opset1::Convert>({q_proj_weight}, {{"destination_type", "f32"}});  //  [4096,4096]
    auto q_proj = makePattern<opset1::MatMul>({input, q_proj_weight_cvt | q_proj_weight},
                                              {{"transpose_a", false}, {"transpose_b", true}});  //  [?,?,4096]
    auto result = q_proj;

    matcher_pass_callback callback = [&](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();

        auto src = pattern_map.at(input);

        auto&& children = src.get_target_inputs();

        if (children.size() < 3) {
            return false;
        }

        OutputVector args = {src};
        OutputVector outputs;
        ov::Shape proj_weight_shape;
        for (auto& child : children) {
            auto mm = dynamic_cast<opset1::MatMul*>(child.get_node());
            if (!mm) {
                // maybe a ShapeOf
                continue;
            }
            if (mm->get_transpose_a() != false || mm->get_transpose_b() != true) {
                return false;
            }
            auto constw = ov::as_type_ptr<opset1::Constant>(mm->input_value(1).get_node_shared_ptr());
            if (!constw) {
                auto cvt = ov::as_type_ptr<opset1::Convert>(mm->input_value(1).get_node_shared_ptr());
                if (!cvt) {
                    return false;
                }
                constw = ov::as_type_ptr<opset1::Constant>(cvt->input_value(0).get_node_shared_ptr());
            }
            if (!constw) {
                return false;
            }

            // make sure all weights are the same
            if (proj_weight_shape.empty()) {
                proj_weight_shape = constw->get_shape();
            } else if (proj_weight_shape != constw->get_shape()) {
                return false;
            }

            args.push_back(constw);
            outputs.push_back(mm->get_default_output());
        }

        // make sure just 3 projections are found
        if (outputs.size() != 3) {
            return false;
        }

        if (proj_weight_shape[0] != proj_weight_shape[1]) {
            return false;
        }

        LLMMLPNode::Config config;
        config.is_qkv_proj = true;
        config.hidden_size = proj_weight_shape[0];
        config.intermediate_size = proj_weight_shape[0] * 3;

        auto old_node = root;
        auto new_node = std::make_shared<LLMMLPNode>(args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({old_node}, new_node);

        for (size_t i = 0; i < outputs.size(); i++) {
            auto target = outputs[i].get_node_shared_ptr();
            outputs[i].replace(new_node->output(i));
            new_node->add_node_control_dependents(target);
            new_node->add_node_control_dependencies(target);
            target->clear_control_dependents();
        }
        std::cout << "QKVProjFusion:" << old_node->get_friendly_name() << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}
