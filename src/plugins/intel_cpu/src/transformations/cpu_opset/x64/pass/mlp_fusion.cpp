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
#include "openvino/opsets/opset7.hpp"
#include "openvino/opsets/opset8.hpp"
#include "openvino/pass/pattern/matcher.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/cpu_opset/x64/op/llm_mlp.hpp"
#include "transformations/utils/utils.hpp"
#include "transformations/utils/gen_pattern.hpp"

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
    auto mlp_gelu_gate = makePattern<opset7::Gelu>({mlp_gate_proj});
    auto mlp_up_proj = makePattern<opset1::MatMul>({input, up_proj_weight | up_proj_weight_compressed},
                                                   {{"transpose_a", false}, {"transpose_b", true}});
    auto mlp_gated_up = makePattern<opset1::Multiply>({mlp_silu_gate | mlp_gelu_gate, mlp_up_proj}, {{"auto_broadcast", "numpy"}});
    auto down_proj = makePattern<opset1::MatMul>({mlp_gated_up, down_proj_weight | down_proj_weight_compressed},
                                                 {{"transpose_a", false}, {"transpose_b", true}});  //  [?,?,down_size]

    auto result = down_proj;

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
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

        //  Limitation: MLP kernel requires K dimension to be multiple of 256
        if ((up_size % 256) != 0) {
            return false;
        }
        if ((down_size % 256) != 0) {
            return false;
        }

        LLMMLPNode::Config config;
        OutputVector new_args;
        config.is_act_silu = pattern_map.count(mlp_silu_gate) > 0;
        config.is_act_gelu = pattern_map.count(mlp_gelu_gate) > 0;
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
                               pattern_map.at(config.is_act_silu ? mlp_silu_gate : mlp_gelu_gate).get_node_shared_ptr(),
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
