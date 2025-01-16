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

    auto input = makePattern("[?,?,?]");

    auto gate_proj_weight_compressed = makePattern<opset1::Constant>({});  // [up_size, down_size]
    auto gate_proj_weight = makePattern<opset1::Convert>({gate_proj_weight_compressed}, {{"destination_type", "f32"}});

    auto up_proj_weight_compressed = makePattern<opset1::Constant>({});  // [up_size, down_size]
    auto up_proj_weight = makePattern<opset1::Convert>({up_proj_weight_compressed}, {{"destination_type", "f32"}});

    auto down_proj_weight_compressed = makePattern<opset1::Constant>({});  // [down_size, up_size]
    auto down_proj_weight = makePattern<opset1::Convert>({down_proj_weight_compressed}, {{"destination_type", "f32"}});

    // symmetrically INT8 quantized version
    // all 3 layers must be quantized at the same time (checked in callback)
    auto gate_proj_weight_i8 =
        makeConst(ov::element::i8, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto gate_proj_weight_scales_per_OC = makeConst(ov::element::f32, ov::PartialShape({ov::Dimension(), 1}), nullptr);
    auto gate_proj_weight_f32 = makePattern<opset1::Convert>({gate_proj_weight_i8}, {{"destination_type", "f32"}});
    auto gate_proj_weight_deq =
        makePattern<opset1::Multiply>({gate_proj_weight_f32, gate_proj_weight_scales_per_OC}, {{"auto_broadcast", "numpy"}});

    auto up_proj_weight_i8 =
        makeConst(ov::element::i8, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto up_proj_weight_scales_per_OC = makeConst(ov::element::f32, ov::PartialShape({ov::Dimension(), 1}), nullptr);
    auto up_proj_weight_f32 = makePattern<opset1::Convert>({up_proj_weight_i8}, {{"destination_type", "f32"}});
    auto up_proj_weight_deq =
        makePattern<opset1::Multiply>({up_proj_weight_f32, up_proj_weight_scales_per_OC}, {{"auto_broadcast", "numpy"}});

    auto down_proj_weight_i8 =
        makeConst(ov::element::i8, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto down_proj_weight_scales_per_OC = makeConst(ov::element::f32, ov::PartialShape({ov::Dimension(), 1}), nullptr);
    auto down_proj_weight_f32 = makePattern<opset1::Convert>({down_proj_weight_i8}, {{"destination_type", "f32"}});
    auto down_proj_weight_deq =
        makePattern<opset1::Multiply>({down_proj_weight_f32, down_proj_weight_scales_per_OC}, {{"auto_broadcast", "numpy"}});

    // gate-up weights are combined
    auto gate_up_proj_weight = makeConst(ov::element::f16, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto gate_up_proj_weight_f32 = makePattern<opset1::Convert>({gate_up_proj_weight}, {{"destination_type", "f32"}});

    auto gate_up_proj_weight_const_i8 =
        makeConst(ov::element::i8, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto gate_up_proj_weight_cvt_f32 = makePattern<opset1::Convert>({gate_up_proj_weight_const_i8}, {{"destination_type", "f32"}});
    auto gate_up_proj_weight_scales_per_OC = makeConst(ov::element::f32, ov::PartialShape({ov::Dimension(), 1}), nullptr);
    auto gate_up_proj_weight_deq = makePattern<opset1::Multiply>({gate_up_proj_weight_cvt_f32, gate_up_proj_weight_scales_per_OC},
                                                             {{"auto_broadcast", "numpy"}});

    auto gate_up_proj = makePattern<opset1::MatMul>({input, gate_up_proj_weight_f32 | gate_up_proj_weight_deq},
                                                    {{"transpose_a", false}, {"transpose_b", true}});
    auto gate_up_split_lengths = makeConst(ov::element::i32,
                                           ov::Shape({
                                               2,
                                           }),
                                           nullptr);
    auto gate_up_proj_split = makePattern<opset1::VariadicSplit>({gate_up_proj, -1, gate_up_split_lengths});
    gate_up_proj_split->set_output_size(2);

    auto mlp_gate_proj = makePattern<opset1::MatMul>({input, gate_proj_weight | gate_proj_weight_compressed | gate_proj_weight_deq},
                                                     {{"transpose_a", false}, {"transpose_b", true}});  // [?,?,up_size]
    auto mlp_silu_gate = makePattern<opset4::Swish>({mlp_gate_proj | gate_up_proj_split->output(0)});
    auto mlp_gelu_gate = makePattern<opset7::Gelu>({mlp_gate_proj | gate_up_proj_split->output(0)});
    auto mlp_up_proj = makePattern<opset1::MatMul>({input, up_proj_weight | up_proj_weight_compressed | up_proj_weight_deq},
                                                   {{"transpose_a", false}, {"transpose_b", true}});

    auto mlp_gated_up =
        makePattern<opset1::Multiply>({mlp_silu_gate | mlp_gelu_gate, mlp_up_proj | gate_up_proj_split->output(1)},
                                      {{"auto_broadcast", "numpy"}});
    auto down_proj = makePattern<opset1::MatMul>({mlp_gated_up, down_proj_weight | down_proj_weight_compressed | down_proj_weight_deq},
                                                 {{"transpose_a", false}, {"transpose_b", true}});  //  [?,?,down_size]

    auto result = down_proj;

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        PatternValidator validator(m);
        if (!validator) {
            return false;
        }

        const auto& pattern_map = m.get_pattern_value_map();
        auto root = m.get_match_root();
        auto src = pattern_map.at(input);
        if (!src.get_element_type().is_real()) {
            // FakeQuantize, should skip fusion
            return false;
        }
        Output<Node> gate_proj_w;
        Output<Node> up_proj_w;
        Output<Node> down_proj_w;

        // down projection is harder to quantize w/o causing accuracy problem, so it may be un-quantized instead
        bool is_gate_up_quantized_int8 = false;
        bool is_down_proj_int8 = false;
        bool is_gate_up_combined = false;
        if (pattern_map.count(gate_up_proj_weight_const_i8) > 0 && pattern_map.count(down_proj_weight_compressed) > 0) {
            //gate-up combined & quantized
            is_gate_up_quantized_int8 = true;
            is_gate_up_combined = true;
            gate_proj_w = pattern_map.at(gate_up_proj_weight_const_i8);
            up_proj_w = pattern_map.at(gate_up_proj_weight_const_i8);
            down_proj_w = pattern_map.at(down_proj_weight_compressed);
        } else if (pattern_map.count(gate_up_proj_weight) > 0 && pattern_map.count(down_proj_weight_compressed) > 0) {
            //gate-up combined
            is_gate_up_combined = true;
            gate_proj_w = pattern_map.at(gate_up_proj_weight);
            up_proj_w = pattern_map.at(gate_up_proj_weight);
            down_proj_w = pattern_map.at(down_proj_weight_compressed);
        } else if (pattern_map.count(gate_proj_weight_compressed) > 0 && pattern_map.count(up_proj_weight_compressed) > 0 &&
            pattern_map.count(down_proj_weight_compressed) > 0) {
            is_gate_up_quantized_int8 = false;
            is_down_proj_int8 = false;
            gate_proj_w = pattern_map.at(gate_proj_weight_compressed);
            up_proj_w = pattern_map.at(up_proj_weight_compressed);
            down_proj_w = pattern_map.at(down_proj_weight_compressed);
        } else if (pattern_map.count(gate_proj_weight_i8) > 0 && pattern_map.count(up_proj_weight_i8) > 0 &&
                   pattern_map.count(gate_proj_weight_scales_per_OC) > 0 && pattern_map.count(up_proj_weight_scales_per_OC) > 0) {
            is_gate_up_quantized_int8 = true;
            gate_proj_w = pattern_map.at(gate_proj_weight_i8);
            up_proj_w = pattern_map.at(up_proj_weight_i8);

            if (pattern_map.count(down_proj_weight_i8) > 0) {
                if (pattern_map.count(down_proj_weight_scales_per_OC) == 0) return false;
                is_down_proj_int8 = true;
                down_proj_w = pattern_map.at(down_proj_weight_i8);
            } else {
                is_down_proj_int8 = false;
                down_proj_w = pattern_map.at(down_proj_weight_compressed);
            }
        } else {
            return false;
        }

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

        auto up_size = is_gate_up_combined ? (up_shape[0] / 2) : (up_shape[0]);
        auto down_size = up_shape[1];
        if (down_shape[0] != down_size)
            return false;
        if (down_shape[1] != up_size)
            return false;

        LLMMLPNode::Config config;
        OutputVector new_args;
        std::shared_ptr<Node> gate_act;

        config.gate_up_quantized = is_gate_up_quantized_int8;
        config.down_quantized = is_down_proj_int8;
        config.hidden_size = down_size;
        config.up_size = up_size;
        config.gate_up_combined = is_gate_up_combined;
        if (pattern_map.count(mlp_silu_gate) > 0) {
            config.act = LLMMLPNode::ACT_FN::SILU;
            gate_act = mlp_silu_gate;
        } else if (pattern_map.count(mlp_gelu_gate) > 0) {
            config.act = LLMMLPNode::ACT_FN::GELU;
            gate_act = mlp_gelu_gate;
        } else {
            return false;
        }

        new_args.push_back(src);
        new_args.push_back(gate_proj_w);
        new_args.push_back(up_proj_w);
        new_args.push_back(down_proj_w);
        if (is_gate_up_quantized_int8) {
            if (is_gate_up_combined) {
                new_args.push_back(pattern_map.at(gate_up_proj_weight_scales_per_OC));
                new_args.push_back(pattern_map.at(gate_up_proj_weight_scales_per_OC));
            } else {
                new_args.push_back(pattern_map.at(gate_proj_weight_scales_per_OC));
                new_args.push_back(pattern_map.at(up_proj_weight_scales_per_OC));
            }
        }
        if (is_down_proj_int8) {
            new_args.push_back(pattern_map.at(down_proj_weight_scales_per_OC));
        }

        auto old_node = root;
        auto new_node = std::make_shared<LLMMLPNode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());
        ov::copy_runtime_info({pattern_map.at(gate_act).get_node_shared_ptr(),
                               pattern_map.at(down_proj).get_node_shared_ptr()},
                              new_node);
        if (is_gate_up_combined) {
            ov::copy_runtime_info({pattern_map.at(gate_up_proj).get_node_shared_ptr()}, new_node);
        } else {
            ov::copy_runtime_info({pattern_map.at(mlp_gate_proj).get_node_shared_ptr(),
                                   pattern_map.at(mlp_up_proj).get_node_shared_ptr()}, new_node);
        }
        // callback is for plugin implementation to check if it can be supported
        if (!transformation_callback(new_node)) {
            return false;
        }

        ov::replace_node(old_node, new_node);
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}
