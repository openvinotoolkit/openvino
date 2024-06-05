// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "qkv_proj_fusion.hpp"

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
#include "transformations/cpu_opset/x64/op/llm_qkv_proj.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/utils/gen_pattern.hpp"

using namespace ov::gen_pattern;

ov::intel_cpu::QKVProjFusion::QKVProjFusion() {
    MATCHER_SCOPE(QKVProjFusion);

    auto input = makePattern("f32[?,?,?]");

    auto q_proj_weight = makePattern<opset1::Constant>({});
    auto q_proj_weight_cvt =
        makePattern<opset1::Convert>({q_proj_weight}, {{"destination_type", "f32"}});  //  [4096,4096]
    auto q_proj = makePattern<opset1::MatMul>({input, q_proj_weight_cvt | q_proj_weight},
                                              {{"transpose_a", false}, {"transpose_b", true}});  //  [?,?,4096]
    auto result = q_proj;

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
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

            // make sure all weights have the same shape
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

        // Limitation: QKV kernels requires K dimension to be multiple of 256
        if ((proj_weight_shape[0] % 256) != 0) {
            return false;
        }

        LLMQKVProjNode::Config config;
        config.hidden_size = proj_weight_shape[0];

        auto old_node = root;
        auto new_node = std::make_shared<LLMQKVProjNode>(args, config);
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
