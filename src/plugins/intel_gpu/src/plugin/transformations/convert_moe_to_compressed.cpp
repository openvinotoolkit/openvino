// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_moe_to_compressed.hpp"

#include <memory>

#include "intel_gpu/op/moe_compressed.hpp"
#include "openvino/core/except.hpp"
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
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
using namespace ov::pass::pattern;

#define MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(SUFFIX)\
    auto compressed_weights_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));\
    auto zp_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));\
    \
    auto weight_convert_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({compressed_weights_m_##SUFFIX}, type_matches(ov::element::f16));\
    auto zp_convert_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({zp_m_##SUFFIX}, type_matches(ov::element::f16));\
    auto sub_m_##SUFFIX = wrap_type<ov::op::v1::Subtract>({weight_convert_m_##SUFFIX, zp_convert_m_##SUFFIX});\
    \
    auto scale_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::f16));\
    auto mul_m_##SUFFIX = wrap_type<ov::op::v1::Multiply>({sub_m_##SUFFIX, scale_m_##SUFFIX});\
    \
    auto reshape_ungroup_##SUFFIX = [](const ov::Output<ov::Node>& output) {\
        auto in_ps = output.get_node()->get_input_partial_shape(0);\
        auto out_ps = output.get_node()->get_output_partial_shape(0);\
        return in_ps.rank().is_static() && out_ps.rank().is_static() && (in_ps.size() == 4 && out_ps.size() == 3);\
    };\
    \
    auto reshape_const_m_##SUFFIX = wrap_type<ov::op::v0::Constant>();\
    auto reshape_m_##SUFFIX = wrap_type<ov::op::v1::Reshape>({mul_m_##SUFFIX, reshape_const_m_##SUFFIX}, reshape_ungroup_##SUFFIX);\
    \
    auto convert_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({reshape_m_##SUFFIX}, type_matches(ov::element::f32));

#define MOE_COMPRESSED_WEIGHT_GEMM3(SUFFIX)\
    auto scale_##SUFFIX = pattern_map.at(scale_m_##SUFFIX).get_node_shared_ptr();\
    auto zp_##SUFFIX = pattern_map.at(zp_m_##SUFFIX).get_node_shared_ptr();\
    auto scale_shape_##SUFFIX = scale_##SUFFIX->get_shape();\
    scale_shape_##SUFFIX.pop_back();\
    auto reshape_const_##SUFFIX = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ scale_shape_##SUFFIX.size() }, scale_shape_##SUFFIX);\
    auto scale_reshape_##SUFFIX = std::make_shared<ov::op::v1::Reshape>(scale_##SUFFIX, reshape_const_##SUFFIX, false);\
    auto zp_reshape_##SUFFIX = std::make_shared<ov::op::v1::Reshape>(zp_##SUFFIX, reshape_const_##SUFFIX, false);\
    \
    std::vector<size_t> transpose_order_##SUFFIX(scale_reshape_##SUFFIX->get_shape().size());\
    std::iota(transpose_order_##SUFFIX.begin(), transpose_order_##SUFFIX.end(), 0);\
    std::swap(*(transpose_order_##SUFFIX.end() - 1), *(transpose_order_##SUFFIX.end() - 2));\
    auto transpose_const_##SUFFIX = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{ transpose_order_##SUFFIX.size() }, transpose_order_##SUFFIX);\
    auto transpose_scale_##SUFFIX = std::make_shared<ov::op::v1::Transpose>(scale_reshape_##SUFFIX, transpose_const_##SUFFIX);\
    auto transpose_zp_##SUFFIX = std::make_shared<ov::op::v1::Transpose>(zp_reshape_##SUFFIX, transpose_const_##SUFFIX);

ConvertMOEToMOECompressed::ConvertMOEToMOECompressed() {
    MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(gate);
    MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(up);
    MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(down);

    auto hidden_states_m = any_input();
    auto routing_weights_m = any_input();
    auto topk_m = any_input();

    auto moe_root = wrap_type<ov::op::internal::MOE>({hidden_states_m, routing_weights_m, topk_m, convert_m_gate, convert_m_up, convert_m_down},
        [](const ov::Output<ov::Node>& output) {
            auto moe = ov::as_type_ptr<ov::op::internal::MOE>(output.get_node_shared_ptr());
            return moe && moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
        });

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto moe = ov::as_type_ptr<ov::op::internal::MOE>(pattern_map.at(moe_root).get_node_shared_ptr());
        if (!moe || transformation_callback(moe)) {
            return false;
        }
        MOE_COMPRESSED_WEIGHT_GEMM3(gate);
        MOE_COMPRESSED_WEIGHT_GEMM3(up);
        MOE_COMPRESSED_WEIGHT_GEMM3(down);

        OutputVector args(12);
        args[0] = pattern_map.at(hidden_states_m);
        args[1] = pattern_map.at(routing_weights_m);
        args[2] = pattern_map.at(topk_m);
        args[3] = pattern_map.at(compressed_weights_m_gate);
        args[4] = transpose_scale_gate;
        args[5] = transpose_zp_gate;
        args[6] = pattern_map.at(compressed_weights_m_up);
        args[7] = transpose_scale_up;
        args[8] = transpose_zp_up;
        args[9] = pattern_map.at(compressed_weights_m_down);
        args[10] = transpose_scale_down;
        args[11] = transpose_zp_down;
        ov::intel_gpu::op::MOECompressed::Config config(moe->get_config());
        auto wei_partial_shape = pattern_map.at(compressed_weights_m_up).get_partial_shape();
        OPENVINO_ASSERT(wei_partial_shape.is_static(), "moe weight shape should be static.");
        auto weight_shape = wei_partial_shape.to_shape();
        if (weight_shape.size() != 4) {
            return false;
        }
        config.hidden_size = weight_shape[2] * weight_shape[3];
        config.inter_size = weight_shape[1];
        config.num_expert = weight_shape[0];
        config.group_size = weight_shape[3];
        auto topk_shape = pattern_map.at(topk_m).get_partial_shape();
        OPENVINO_ASSERT(topk_shape[1].is_static(), "k dimenion in moe topk input should be static.");
        config.top_k = topk_shape[1].get_length();
        config.out_type = ov::element::f16;
        auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(args, config);

        moe_compressed->set_friendly_name(moe->get_friendly_name());
        ov::copy_runtime_info(moe, moe_compressed);
        ov::replace_node(moe, moe_compressed);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_root, "ConvertMOEToMOECompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu
