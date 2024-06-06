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
        size_t hidden_size = 0;
        std::vector<int> proj_size;
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

            // input feature size should be the same
            const auto& wshape = constw->get_shape();
            if (hidden_size == 0) {
                hidden_size = wshape[1];
            } else if (hidden_size != wshape[1]) {
                return false;
            }

            // AMX tiles limit, tail handling is not supported yet
            if ((wshape[0] % 32) != 0)
                return false;
            proj_size.push_back(wshape[0]);
            args.push_back(constw);
            outputs.push_back(mm->get_default_output());
        }

        LLMQKVProjNode::Config config;

        // make sure just 3 projections are found
        if (outputs.size() != 3) {
            return false;
        }
        if (args.size() != 4) {
            return false;
        }

        config.hidden_size = static_cast<int>(hidden_size);

        // Limitation: QKV kernels requires K dimension to be multiple of 256
        if ((config.hidden_size % 256) != 0)
            return false;

        // proj size of q/k/v can be different (considering multi-query)
        config.proj_size0 = proj_size[0];
        config.proj_size1 = proj_size[1];
        config.proj_size2 = proj_size[2];

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
