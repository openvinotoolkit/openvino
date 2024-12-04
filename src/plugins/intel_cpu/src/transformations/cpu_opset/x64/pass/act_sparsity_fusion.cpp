// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "act_sparsity_fusion.hpp"

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
#include "transformations/cpu_opset/x64/op/act_sparse_fc.hpp"
#include "transformations/utils/gen_pattern.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::gen_pattern;

ov::intel_cpu::ActivationSparsityFusion::ActivationSparsityFusion() {
    MATCHER_SCOPE(ActivationSparsityFusion);

    auto input = makePattern("[?,?,?]");

    auto Abs_input = makePattern<opset1::Abs>({input});
    auto sparsity_threshold = makeConst(element::f32,
                                        ov::Shape({
                                            1,
                                            1,
                                            1,
                                        }),
                                        nullptr);
    auto LessEqual = makePattern<ov::op::TypeRelaxed<opset1::LessEqual>>({Abs_input, sparsity_threshold},
                                                                         {{"auto_broadcast", "numpy"}});
    auto sparse_input =
        makePattern<ov::op::TypeRelaxed<opset1::Select>>({LessEqual, 0.000000f, input}, {{"auto_broadcast", "numpy"}});

    auto fc_weight_compressed = makePattern<opset1::Constant>({});
    auto fc_weight = makePattern<opset1::Convert>({fc_weight_compressed}, {{"destination_type", "f32"}});

    // symmetrically INT8 quantized version
    // all 3 layers must be quantized at the same time (checked in callback)
    auto fc_weight_i8 = makeConst(ov::element::i8, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto fc_weight_u8 = makeConst(ov::element::u8, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto fc_weight_f32 = makePattern<opset1::Convert>({fc_weight_i8 | fc_weight_u8}, {{"destination_type", "f32"}});

    auto fc_weight_zero_point_u8 =
        makeConst(ov::element::u8, ov::PartialShape({ov::Dimension(), ov::Dimension()}), nullptr);
    auto fc_weight_zero_point_f32 =
        makePattern<opset1::Convert>({fc_weight_zero_point_u8}, {{"destination_type", "f32"}});

    auto fc_weight_zp =
        makePattern<opset1::Subtract>({fc_weight_f32, fc_weight_zero_point_f32}, {{"auto_broadcast", "numpy"}});

    auto fc_weight_scales_per_OC = makeConst(ov::element::f32, ov::PartialShape({ov::Dimension(), 1}), nullptr);
    auto fc_weight_deq =
        makePattern<opset1::Multiply>({fc_weight_zp, fc_weight_scales_per_OC}, {{"auto_broadcast", "numpy"}});


    // INT4 groupped quantization would have a reshape
    //      weight const  u4  [OC, IC//group_size, group_size]
    //           convert f16  [OC, IC//group_size, group_size]
    //  zero_point const  u4  [OC, IC//group_size, 1]
    //           convert f16  [OC, IC//group_size, 1]
    //          Subtract f16  [OC, IC//group_size, group_size]
    //       scale const f16  [OC, IC//group_size, 1]
    //          Multiply f16  [OC, IC//group_size, group_size] x [OC, IC//group_size, 1]
    //           Reshape f16  [OC, IC//group_size, group_size] => [OC, IC]
    //
    auto fc_weight_u4 =
        makeConst(ov::element::u4, ov::PartialShape({ov::Dimension(), ov::Dimension(), ov::Dimension()}), nullptr);
    auto fc_weight_u4f32 = makePattern<opset1::Convert>({fc_weight_u4}, {{"destination_type", "f32"}});
    auto fc_weight_zero_point_u4 =
        makeConst(ov::element::u4, ov::PartialShape({ov::Dimension(), ov::Dimension(), ov::Dimension(1)}), nullptr);
    auto fc_weight_zero_point_u4f32 =
        makePattern<opset1::Convert>({fc_weight_zero_point_u4}, {{"destination_type", "f32"}});
    auto fc_weight_sub_zp =
        makePattern<opset1::Subtract>({fc_weight_u4f32, fc_weight_zero_point_u4f32}, {{"auto_broadcast", "numpy"}});
    auto fc_weight_scales_gr =
        makeConst(ov::element::f32, ov::PartialShape({ov::Dimension(), ov::Dimension(), ov::Dimension(1)}), nullptr);
    auto fc_weight_u4_deq =
        makePattern<opset1::Multiply>({fc_weight_sub_zp, fc_weight_scales_gr}, {{"auto_broadcast", "numpy"}});
    auto fc_weight_shape = makePattern("[?]");
    auto fc_weight_u4_deq_reshape =
        makePattern<opset1::Reshape>({fc_weight_u4_deq, fc_weight_shape}, {{"special_zero", false}});

    auto fc_result = makePattern<opset1::MatMul>(
        {sparse_input, fc_weight | fc_weight_compressed | fc_weight_deq | fc_weight_u4_deq_reshape},
        {{"transpose_a", false}, {"transpose_b", true}});  // [?,?,up_size]

    auto result = fc_result;

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

        OutputVector new_args;

        new_args.push_back(pattern_map.at(input));

        ActSparseFCNode::Config config;
        config.ic = 0;
        config.oc = 0;
        config.ic_q_group_size = 0; // per-OC by default
        config.is_int4 = false;

        if (pattern_map.count(fc_weight_u8) > 0) {
            new_args.push_back(pattern_map.at(fc_weight_u8));
            new_args.push_back(pattern_map.at(fc_weight_zero_point_u8));
            new_args.push_back(pattern_map.at(fc_weight_scales_per_OC));
            auto const_weight = ov::as_type_ptr<opset1::Constant>(pattern_map.at(fc_weight_u8).get_node_shared_ptr());
            if (!const_weight)
                return false;
            const auto& w_shape = const_weight->get_shape();
            config.oc = w_shape[0];
            config.ic = w_shape[1];
        } else if (pattern_map.count(fc_weight_u4) > 0) {
            new_args.push_back(pattern_map.at(fc_weight_u4));
            new_args.push_back(pattern_map.at(fc_weight_zero_point_u4));
            new_args.push_back(pattern_map.at(fc_weight_scales_gr));

            auto const_weight = ov::as_type_ptr<opset1::Constant>(pattern_map.at(fc_weight_u4).get_node_shared_ptr());
            if (!const_weight)
                return false;

            const auto& w_shape = const_weight->get_shape();
            config.oc = w_shape[0];
            config.ic = w_shape[1] * w_shape[2];
            config.ic_q_group_size = w_shape[2];
            config.is_int4 = true;
        } else {
            return false;
        }

        auto const_thr = ov::as_type_ptr<opset1::Constant>(pattern_map.at(sparsity_threshold).get_node_shared_ptr());
        if (!const_thr)
            return false;

        auto thr = const_thr->get_vector<float>();
        if (thr.size() != 1)
            return false;

        config.threshold = thr[0];

        auto old_node = root;
        auto new_node = std::make_shared<ActSparseFCNode>(new_args, config);
        new_node->set_friendly_name(old_node->get_friendly_name());

        // callback is for plugin implementation to check if it can be supported
        if (!transformation_callback(new_node)) {
            return false;
        }

        if (std::getenv("NO_SPARSE"))
            return false;
        ov::replace_node(old_node, new_node);

        std::cout << m.get_match_root() << std::endl;

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(result, matcher_name);
    this->register_matcher(m, callback);
}
