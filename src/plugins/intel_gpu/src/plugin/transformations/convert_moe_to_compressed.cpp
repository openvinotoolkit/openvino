// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_moe_to_compressed.hpp"

#include <memory>

#include "intel_gpu/op/moe_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

ConvertMOEToMOECompressed::ConvertMOEToMOECompressed() {
    auto reshape_ungroup = [](const ov::Output<ov::Node>& output) {
        auto in_ps = output.get_node()->get_input_partial_shape(0);
        auto out_ps = output.get_node()->get_output_partial_shape(0);
        return in_ps.rank().is_static() && out_ps.rank().is_static() && (in_ps.size() == 4 && out_ps.size() == 3);
    };
    // first proj
    auto compressed_weights_m_0 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto zp_m_0 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));

    auto weight_convert_const_m_0 = wrap_type<ov::op::v0::Convert>({compressed_weights_m_0}, type_matches(ov::element::f16));
    auto zp_convert_const_m_0 = wrap_type<ov::op::v0::Convert>({zp_m_0}, type_matches(ov::element::f16));
    auto sub_m_0 = wrap_type<ov::op::v1::Subtract>({weight_convert_const_m_0, zp_convert_const_m_0});

    auto scale_m_0 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::f16));
    auto mul_m_0 = wrap_type<ov::op::v1::Multiply>({sub_m_0, scale_m_0});

    auto reshape_const_m_0 = wrap_type<ov::op::v0::Constant>();
    auto reshape_m_0 = wrap_type<ov::op::v1::Reshape>({mul_m_0, reshape_const_m_0}, reshape_ungroup);

    auto convert_m_0 = wrap_type<ov::op::v0::Convert>({reshape_m_0}, type_matches(ov::element::f32));

    // second proj
    auto compressed_weights_m_1 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto zp_m_1 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));

    auto weight_convert_const_m_1 = wrap_type<ov::op::v0::Convert>({compressed_weights_m_1}, type_matches(ov::element::f16));
    auto zp_convert_const_m_1 = wrap_type<ov::op::v0::Convert>({zp_m_1}, type_matches(ov::element::f16));
    auto sub_m_1 = wrap_type<ov::op::v1::Subtract>({weight_convert_const_m_1, zp_convert_const_m_1});

    auto scale_m_1 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::f16));
    auto mul_m_1 = wrap_type<ov::op::v1::Multiply>({sub_m_1, scale_m_1});

    auto reshape_const_m_1 = wrap_type<ov::op::v0::Constant>();
    auto reshape_m_1 = wrap_type<ov::op::v1::Reshape>({mul_m_1, reshape_const_m_1}, reshape_ungroup);

    auto convert_m_1 = wrap_type<ov::op::v0::Convert>({reshape_m_1}, type_matches(ov::element::f32));

    // third proj
    auto compressed_weights_m_2 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));
    auto zp_m_2 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));

    auto weight_convert_const_m_2 = wrap_type<ov::op::v0::Convert>({compressed_weights_m_2}, type_matches(ov::element::f16));
    auto zp_convert_const_m_2 = wrap_type<ov::op::v0::Convert>({zp_m_2}, type_matches(ov::element::f16));
    auto sub_m_2 = wrap_type<ov::op::v1::Subtract>({weight_convert_const_m_2, zp_convert_const_m_2});

    auto scale_m_2 = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::f16));
    auto mul_m_2 = wrap_type<ov::op::v1::Multiply>({sub_m_2, scale_m_2});

    auto reshape_const_m_2 = wrap_type<ov::op::v0::Constant>();
    auto reshape_m_2 = wrap_type<ov::op::v1::Reshape>({mul_m_2, reshape_const_m_2}, reshape_ungroup);

    auto convert_m_2 = wrap_type<ov::op::v0::Convert>({reshape_m_2}, type_matches(ov::element::f32));

    auto hidden_states_m = any_input();
    auto routing_weights_m = any_input();
    auto topk_m = any_input();

    auto moe_root = wrap_type<ov::op::internal::MOE>({hidden_states_m, routing_weights_m, topk_m, convert_m_0, convert_m_1, convert_m_2},
                                                     [](const ov::Output<ov::Node>& output) {
                                                         auto moe = ov::as_type_ptr<ov::op::internal::MOE>(output.get_node_shared_ptr());
                                                         return moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
                                                     });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto moe = ov::as_type_ptr<ov::op::internal::MOE>(pattern_map.at(moe_root).get_node_shared_ptr());
        if (!moe || transformation_callback(moe)) {
            return false;
        }

        auto input0 = moe->input_value(0);
        auto input1 = moe->input_value(1);
        auto input2 = moe->input_value(2);
        auto input3 = moe->input_value(3);
        auto input4 = moe->input_value(4);
        auto input5 = moe->input_value(5);

        // first proj
        auto scale_0 = pattern_map.at(scale_m_0).get_node_shared_ptr();
        auto zp_0 = pattern_map.at(zp_m_0).get_node_shared_ptr();
        auto scale_0_shape = scale_0->get_shape();
        scale_0_shape.pop_back();
        auto reshape_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{scale_0_shape.size()}, scale_0_shape);
        auto scale_0_reshape = std::make_shared<ov::op::v1::Reshape>(scale_0, reshape_const, false);
        auto zp_0_reshape = std::make_shared<ov::op::v1::Reshape>(zp_0, reshape_const, false);
        ov::enable_keep_const_precision(scale_0_reshape);
        ov::enable_keep_const_precision(zp_0_reshape);

        std::vector<size_t> transpose_order_0(scale_0_reshape->get_shape().size());
        std::iota(transpose_order_0.begin(), transpose_order_0.end(), 0);
        std::swap(*(transpose_order_0.end() - 1), *(transpose_order_0.end() - 2));
        auto transpose_0_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{transpose_order_0.size()}, transpose_order_0);
        auto transpose_0_scale = std::make_shared<ov::op::v1::Transpose>(scale_0_reshape, transpose_0_const);
        auto transpose_0_zp = std::make_shared<ov::op::v1::Transpose>(zp_0_reshape, transpose_0_const);
        ov::enable_keep_const_precision(transpose_0_scale);
        ov::enable_keep_const_precision(transpose_0_zp);

        // second proj
        auto scale_1 = pattern_map.at(scale_m_1).get_node_shared_ptr();
        auto zp_1 = pattern_map.at(zp_m_1).get_node_shared_ptr();
        auto scale_1_shape = scale_1->get_shape();
        scale_1_shape.pop_back();
        auto reshape_const_1 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{scale_1_shape.size()}, scale_1_shape);
        auto scale_1_reshape = std::make_shared<ov::op::v1::Reshape>(scale_1, reshape_const_1, false);
        auto zp_1_reshape = std::make_shared<ov::op::v1::Reshape>(zp_1, reshape_const_1, false);
        ov::enable_keep_const_precision(scale_1_reshape);
        ov::enable_keep_const_precision(zp_1_reshape);

        std::vector<size_t> transpose_order_1(scale_1_reshape->get_shape().size());
        std::iota(transpose_order_1.begin(), transpose_order_1.end(), 0);
        std::swap(*(transpose_order_1.end() - 1), *(transpose_order_1.end() - 2));
        auto transpose_1_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{transpose_order_1.size()}, transpose_order_1);
        auto transpose_1_scale = std::make_shared<ov::op::v1::Transpose>(scale_1_reshape, transpose_1_const);
        auto transpose_1_zp = std::make_shared<ov::op::v1::Transpose>(zp_1_reshape, transpose_1_const);
        ov::enable_keep_const_precision(transpose_1_scale);
        ov::enable_keep_const_precision(transpose_1_zp);

        // third proj
        auto scale_2 = pattern_map.at(scale_m_2).get_node_shared_ptr();
        auto zp_2 = pattern_map.at(zp_m_2).get_node_shared_ptr();
        auto scale_2_shape = scale_2->get_shape();
        scale_2_shape.pop_back();
        auto reshape_const_2 = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{scale_2_shape.size()}, scale_2_shape);
        auto scale_2_reshape = std::make_shared<ov::op::v1::Reshape>(scale_2, reshape_const_2, false);
        auto zp_2_reshape = std::make_shared<ov::op::v1::Reshape>(zp_2, reshape_const_2, false);
        ov::enable_keep_const_precision(scale_2_reshape);
        ov::enable_keep_const_precision(zp_2_reshape);

        std::vector<size_t> transpose_order_2(scale_2_reshape->get_shape().size());
        std::iota(transpose_order_2.begin(), transpose_order_2.end(), 0);
        std::swap(*(transpose_order_2.end() - 1), *(transpose_order_2.end() - 2));
        auto transpose_2_const = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{transpose_order_2.size()}, transpose_order_2);
        auto transpose_2_scale = std::make_shared<ov::op::v1::Transpose>(scale_2_reshape, transpose_2_const);
        auto transpose_2_zp = std::make_shared<ov::op::v1::Transpose>(zp_2_reshape, transpose_2_const);
        ov::enable_keep_const_precision(transpose_2_scale);
        ov::enable_keep_const_precision(transpose_2_zp);

        OutputVector args(12);
        args[0] = pattern_map.at(hidden_states_m);
        args[1] = pattern_map.at(routing_weights_m);
        args[2] = pattern_map.at(topk_m);
        args[3] = pattern_map.at(compressed_weights_m_0);
        args[4] = transpose_0_scale;
        args[5] = transpose_0_zp;
        args[6] = pattern_map.at(compressed_weights_m_1);
        args[7] = transpose_1_scale;
        args[8] = transpose_1_zp;
        args[9] = pattern_map.at(compressed_weights_m_2);
        args[10] = transpose_2_scale;
        args[11] = transpose_2_zp;
        ov::intel_gpu::op::MOECompressed::Config config;
        auto weight_shape = pattern_map.at(compressed_weights_m_0).get_shape();
        if (weight_shape.size() != 4) {
            return false;
        }
        auto topk_shape = pattern_map.at(topk_m).get_partial_shape();
        config.hidden_size = weight_shape[2] * weight_shape[3];
        config.inter_size = weight_shape[1];
        config.num_expert = weight_shape[0];
        config.group_size = weight_shape[3];
        config.top_k = topk_shape[1].get_length();
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(args, config);

        auto w0 = pattern_map.at(compressed_weights_m_0).get_node_shared_ptr();
        auto s0 = pattern_map.at(scale_m_0).get_node_shared_ptr();
        auto z0 = pattern_map.at(zp_m_0).get_node_shared_ptr();
        auto w1 = pattern_map.at(compressed_weights_m_1).get_node_shared_ptr();
        auto s1 = pattern_map.at(scale_m_1).get_node_shared_ptr();
        auto z1 = pattern_map.at(zp_m_1).get_node_shared_ptr();
        auto w2 = pattern_map.at(compressed_weights_m_2).get_node_shared_ptr();
        auto s2 = pattern_map.at(scale_m_2).get_node_shared_ptr();
        auto z2 = pattern_map.at(zp_m_2).get_node_shared_ptr();
        ov::enable_keep_const_precision(w0);
        ov::enable_keep_const_precision(s0);
        ov::enable_keep_const_precision(z0);
        ov::enable_keep_const_precision(w1);
        ov::enable_keep_const_precision(s1);
        ov::enable_keep_const_precision(z1);
        ov::enable_keep_const_precision(w2);
        ov::enable_keep_const_precision(s2);
        ov::enable_keep_const_precision(z2);

        ov::enable_keep_const_precision(moe_compressed->input_value(5).get_node_shared_ptr());
        ov::enable_keep_const_precision(moe_compressed->input_value(8).get_node_shared_ptr());
        ov::enable_keep_const_precision(moe_compressed->input_value(11).get_node_shared_ptr());

        moe_compressed->set_friendly_name(moe->get_friendly_name());
        ov::copy_runtime_info(moe, moe_compressed);
        ov::replace_node(moe, moe_compressed);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_root, "ConvertMOEToMOECompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
