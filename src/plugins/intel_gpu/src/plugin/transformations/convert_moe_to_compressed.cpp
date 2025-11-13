// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "convert_moe_to_compressed.hpp"

#include <limits>
#include <memory>

#include "intel_gpu/op/moe_compressed.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/rt_info/keep_const_precision.hpp"
#include "transformations/utils/utils.hpp"

namespace ov::intel_gpu {
    using namespace ov::pass::pattern;
#define MOE_COMPRESSED_WEIGHT_PATTERN(SUFFIX)\
    auto compressed_constant_##SUFFIX = [](const ov::Output<ov::Node>& output) {\
        return (output.get_element_type() == ov::element::u8 || output.get_element_type() == ov::element::i8 ||\
                output.get_element_type() == ov::element::u4 || output.get_element_type() == ov::element::i4);\
    };\
    \
    auto reshape_squeeze_##SUFFIX = [](const ov::Output<ov::Node>& output) {\
        auto in_ps = output.get_node()->get_input_partial_shape(0);\
        auto out_ps = output.get_node()->get_output_partial_shape(0);\
        return in_ps.rank().is_static() && out_ps.rank().is_static() &&\
        ((in_ps.size() == 3 && out_ps.size() == 2) || (in_ps.size() == 4 && out_ps.size() == 3));\
    };\
    \
    auto compressed_weights_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(compressed_constant_##SUFFIX);\
    auto convert_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({compressed_weights_m_##SUFFIX});\
    \
    auto sub_const_m_##SUFFIX = wrap_type<ov::op::v0::Constant>();\
    auto sub_convert_const_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({sub_const_m_##SUFFIX});\
    auto sub_with_convert_m_##SUFFIX = wrap_type<ov::op::v1::Subtract>({convert_m_##SUFFIX, sub_convert_const_m_##SUFFIX});\
    auto sub_no_convert_m_##SUFFIX = wrap_type<ov::op::v1::Subtract>({convert_m_##SUFFIX, sub_const_m_##SUFFIX});\
    auto subtract_m_##SUFFIX = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{sub_with_convert_m_##SUFFIX, sub_no_convert_m_##SUFFIX});\
    \
    auto mul_const_m_##SUFFIX = wrap_type<ov::op::v0::Constant>();\
    auto mul_with_sub_m_##SUFFIX = wrap_type<ov::op::v1::Multiply>({subtract_m_##SUFFIX, mul_const_m_##SUFFIX});\
    auto mul_no_sub_m_##SUFFIX = wrap_type<ov::op::v1::Multiply>({convert_m_##SUFFIX, mul_const_m_##SUFFIX});\
    auto mul_m_##SUFFIX = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{mul_with_sub_m_##SUFFIX, mul_no_sub_m_##SUFFIX});\
    \
    auto reshape_const_m_##SUFFIX = wrap_type<ov::op::v0::Constant>();\
    auto reshape_m_##SUFFIX = wrap_type<ov::op::v1::Reshape>({mul_m_##SUFFIX, reshape_const_m_##SUFFIX}, reshape_squeeze_##SUFFIX);\
    auto convert_reshape_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({reshape_m_##SUFFIX});\
    \
    auto mul2_const_m_##SUFFIX = wrap_type<ov::op::v0::Constant>();\
    auto mul2_m_##SUFFIX = wrap_type<ov::op::v1::Multiply>({reshape_m_##SUFFIX, mul2_const_m_##SUFFIX});\
    \
    auto transpose_input_##SUFFIX = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{reshape_m_##SUFFIX, mul_m_##SUFFIX});\
    auto transpose_const_m_##SUFFIX = wrap_type<ov::op::v0::Constant>();\
    auto transpose_m_##SUFFIX = wrap_type<ov::op::v1::Transpose>({transpose_input_##SUFFIX, transpose_const_m_##SUFFIX});\
    \
    auto convert_mul_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({mul_m_##SUFFIX});\
    auto compressed_weights_input_m_##SUFFIX =\
    std::make_shared<ov::pass::pattern::op::Or>(ov::OutputVector{reshape_m_##SUFFIX, convert_reshape_m_##SUFFIX, transpose_m_##SUFFIX, \
         mul_m_##SUFFIX, mul2_m_##SUFFIX, convert_mul_m_##SUFFIX});

#define MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(SUFFIX)\
    auto gemm3_compressed_weights_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));\
    auto gemm3_zp_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::u4));\
    \
    auto gemm3_weight_convert_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({gemm3_compressed_weights_m_##SUFFIX}, type_matches(ov::element::f16));\
    auto gemm3_zp_convert_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({gemm3_zp_m_##SUFFIX}, type_matches(ov::element::f16));\
    auto gemm3_sub_m_##SUFFIX = wrap_type<ov::op::v1::Subtract>({gemm3_weight_convert_m_##SUFFIX, gemm3_zp_convert_m_##SUFFIX});\
    \
    auto gemm3_scale_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches(ov::element::f16));\
    auto gemm3_mul_m_##SUFFIX = wrap_type<ov::op::v1::Multiply>({gemm3_sub_m_##SUFFIX, gemm3_scale_m_##SUFFIX});\
    \
    auto gemm3_reshape_ungroup_##SUFFIX = [](const ov::Output<ov::Node>& output) {\
        auto in_ps = output.get_node()->get_input_partial_shape(0);\
        auto out_ps = output.get_node()->get_output_partial_shape(0);\
        return in_ps.rank().is_static() && out_ps.rank().is_static() && (in_ps.size() == 4 && out_ps.size() == 3);\
    };\
    \
    auto gemm3_reshape_const_m_##SUFFIX = wrap_type<ov::op::v0::Constant>();\
    auto gemm3_reshape_m_##SUFFIX = optional<ov::op::v1::Reshape>({gemm3_mul_m_##SUFFIX, gemm3_reshape_const_m_##SUFFIX}, gemm3_reshape_ungroup_##SUFFIX);\
    \
    auto gemm3_convert_m_##SUFFIX = wrap_type<ov::op::v0::Convert>({gemm3_reshape_m_##SUFFIX}, type_matches(ov::element::f32));

#define MOE_COMPRESSED_WEIGHT_GEMM3(SUFFIX)\
    auto gemm3_scale_##SUFFIX = pattern_map.at(gemm3_scale_m_##SUFFIX).get_node_shared_ptr();\
    auto gemm3_zp_##SUFFIX = pattern_map.at(gemm3_zp_m_##SUFFIX).get_node_shared_ptr();\
    auto gemm3_scale_shape_##SUFFIX = gemm3_scale_##SUFFIX->get_shape();\
    gemm3_scale_shape_##SUFFIX.pop_back();\
    auto gemm3_reshape_const_##SUFFIX = ov::op::v0::Constant::create(\
        ov::element::i32, \
        ov::Shape{ gemm3_scale_shape_##SUFFIX.size() }, \
        gemm3_scale_shape_##SUFFIX);\
    auto gemm3_scale_reshape_##SUFFIX = std::make_shared<ov::op::v1::Reshape>(gemm3_scale_##SUFFIX, gemm3_reshape_const_##SUFFIX, false);\
    auto gemm3_zp_reshape_##SUFFIX = std::make_shared<ov::op::v1::Reshape>(gemm3_zp_##SUFFIX, gemm3_reshape_const_##SUFFIX, false);\
    \
    std::vector<size_t> gemm3_transpose_order_##SUFFIX(gemm3_scale_reshape_##SUFFIX->get_shape().size());\
    std::iota(gemm3_transpose_order_##SUFFIX.begin(), gemm3_transpose_order_##SUFFIX.end(), 0);\
    std::swap(*(gemm3_transpose_order_##SUFFIX.end() - 1), *(gemm3_transpose_order_##SUFFIX.end() - 2));\
    auto gemm3_transpose_const_##SUFFIX = ov::op::v0::Constant::create(\
        ov::element::i32, \
        ov::Shape{ gemm3_transpose_order_##SUFFIX.size() }, \
        gemm3_transpose_order_##SUFFIX);\
    auto gemm3_transpose_scale_##SUFFIX = std::make_shared<ov::op::v1::Transpose>(gemm3_scale_reshape_##SUFFIX, gemm3_transpose_const_##SUFFIX);\
    auto gemm3_transpose_zp_##SUFFIX = std::make_shared<ov::op::v1::Transpose>(gemm3_zp_reshape_##SUFFIX, gemm3_transpose_const_##SUFFIX);

ConvertMOEToMOECompressed::ConvertMOEToMOECompressed(bool is_pa) {
    // gemm3 pattern start
    MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(gate);
    MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(up);
    MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(down);

    auto hidden_states_m = any_input();
    auto routing_weights_m = any_input();
    auto topk_m = any_input();

    auto moe_root_gemm3 =
        wrap_type<ov::op::internal::MOE>({hidden_states_m, routing_weights_m, topk_m, gemm3_convert_m_gate, gemm3_convert_m_up, gemm3_convert_m_down},
                                         [](const ov::Output<ov::Node>& output) {
                                             auto moe = ov::as_type_ptr<ov::op::internal::MOE>(output.get_node_shared_ptr());
                                             return moe && moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
                                         });
    // gemm3 pattern finished
    // =========================================================================================
    // gemm2 pattern start
    auto topk_input = any_input();

    auto input_gemm2_m = any_input();
    MOE_COMPRESSED_WEIGHT_PATTERN(up)
    auto bias_up_gemm2_m = any_input();
    MOE_COMPRESSED_WEIGHT_PATTERN(down)
    auto bias_down_gemm2_m = any_input();

    auto topk_gemm2_m = wrap_type<ov::op::v11::TopK>({topk_input, any_input()});
    auto topk_indices_gemm2_m = wrap_type<ov::op::v0::Convert>({topk_gemm2_m});

    auto topk_weight_softmax_m = wrap_type<ov::op::v8::Softmax>({topk_gemm2_m});
    auto topk_weight_slice_m = wrap_type<ov::op::v8::Slice>({topk_weight_softmax_m, any_input(), any_input(), any_input(), any_input()});
    auto topk_weight_scatter_elements_update_m = wrap_type<ov::op::v12::ScatterElementsUpdate>({any_input(), any_input(), topk_weight_slice_m, any_input()});
    auto topk_weight_transpose_m = wrap_type<ov::op::v1::Transpose>({topk_weight_scatter_elements_update_m, any_input()});
    auto topk_weight_reshape_m = wrap_type<ov::op::v1::Reshape>({topk_weight_transpose_m, any_input()});
    auto topk_weight_m = wrap_type<ov::op::v0::Unsqueeze>({topk_weight_reshape_m, any_input()});

    auto moe_root_gemm2 =
        wrap_type<ov::op::internal::MOE>({input_gemm2_m,
                                          topk_weight_m,
                                          topk_indices_gemm2_m,
                                          compressed_weights_input_m_up,
                                          bias_up_gemm2_m,
                                          compressed_weights_input_m_down,
                                          bias_down_gemm2_m},
                                         [](const ov::Output<ov::Node>& output) {
                                             auto moe_op = ov::as_type_ptr<ov::op::internal::MOE>(output.get_node_shared_ptr());
                                             return moe_op->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP;
                                         });

    // gemm2 pattern finished

    auto moe_root = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{moe_root_gemm3, moe_root_gemm2});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        std::cout << "ConvertMOEToMOECompressed|Pattern Matched|" << std::endl;
        auto moe = ov::as_type_ptr<ov::op::internal::MOE>(pattern_map.at(moe_root).get_node_shared_ptr());
        if (!moe || transformation_callback(moe)) {
            return false;
        }
        if (moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU) {
            auto wei_partial_shape = pattern_map.at(gemm3_compressed_weights_m_up).get_partial_shape();
            if (!wei_partial_shape.is_static()) {
                OPENVINO_THROW("Moe weight shape should be static.");
            }
            auto weight_shape = wei_partial_shape.to_shape();
            bool group_compressed = false;
            if (weight_shape.size() == 4) {
                group_compressed = true;
            } else if (weight_shape.size() == 3) {
                group_compressed = false;
            } else {
                OPENVINO_THROW("Moe weight shape must be 3D or 4D.");
            }
            OutputVector args(12);
            args[0] = pattern_map.at(hidden_states_m);
            args[1] = pattern_map.at(routing_weights_m);
            args[2] = pattern_map.at(topk_m);
            args[3] = pattern_map.at(gemm3_compressed_weights_m_gate);
            if (group_compressed) {
                MOE_COMPRESSED_WEIGHT_GEMM3(gate);
                args[4] = gemm3_transpose_scale_gate;
                args[5] = gemm3_transpose_zp_gate;
            } else {
                args[4] = pattern_map.at(gemm3_scale_m_gate);
                args[5] = pattern_map.at(gemm3_zp_m_gate);
            }
            args[6] = pattern_map.at(gemm3_compressed_weights_m_up);
            if (group_compressed) {
                MOE_COMPRESSED_WEIGHT_GEMM3(up);
                args[7] =  gemm3_transpose_scale_up;
                args[8] =  gemm3_transpose_zp_up;
            } else {
                args[7] = pattern_map.at(gemm3_scale_m_up);
                args[8] = pattern_map.at(gemm3_zp_m_up);
            }
            args[9] = pattern_map.at(gemm3_compressed_weights_m_down);
            if (group_compressed) {
                MOE_COMPRESSED_WEIGHT_GEMM3(down);
                args[10] =  gemm3_transpose_scale_down;
                args[11] =  gemm3_transpose_zp_down;
            } else {
                args[10] = pattern_map.at(gemm3_scale_m_down);
                args[11] = pattern_map.at(gemm3_zp_m_down);
            }
            ov::intel_gpu::op::MOECompressed::Config config(moe->get_config());
            config.hidden_size = group_compressed ? weight_shape[2] * weight_shape[3] : weight_shape[2];
            config.inter_size = weight_shape[1];
            config.num_expert = weight_shape[0];
            config.group_size = group_compressed ? weight_shape[3] : std::numeric_limits<size_t>::max();
            auto topk_shape = pattern_map.at(topk_m).get_partial_shape();
            if (!topk_shape[1].is_static()) {
                OPENVINO_THROW("K dimenion in moe topk input should be static..");
            }
            config.top_k = topk_shape[1].get_length();
            config.out_type = ov::element::f16;
            auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(args, config);

            moe_compressed->set_friendly_name(moe->get_friendly_name());
            ov::copy_runtime_info(moe, moe_compressed);
            ov::replace_node(moe, moe_compressed);
        } else if (moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP) {
            OutputVector args;
            auto topk_indice_node = pattern_map.at(topk_indices_gemm2_m);
            auto weight_up_node = pattern_map.at(compressed_weights_input_m_up);
            auto weight_down_node = pattern_map.at(compressed_weights_input_m_down);
            auto topk_node = pattern_map.at(topk_gemm2_m).get_node_shared_ptr();
            auto topk_weight_output = topk_node->output(0);
            auto topk_weight_softmax = pattern_map.at(topk_weight_softmax_m).get_node_shared_ptr();

            auto topk_shape = pattern_map.at(topk_gemm2_m).get_node_shared_ptr()->get_output_partial_shape(1);
            auto weight_shape = pattern_map.at(compressed_weights_input_m_up).get_shape();
            auto weight_down_shape = pattern_map.at(compressed_weights_input_m_down).get_shape();
            auto bias_shape = pattern_map.at(bias_up_gemm2_m).get_node_shared_ptr()->get_shape();
            auto scale_shape = pattern_map.at(mul_const_m_up).get_shape();
            // Weight, scale, zp are assumed to be transposed
            // W     : [num_experts, N, K]
            // scale : [num_experts, N, K / group_size, 1]
            ov::intel_gpu::op::MOECompressed::Config config(moe->get_config());
            config.num_expert = weight_shape[0];
            config.hidden_size = weight_shape[2];
            if (weight_shape.size() == 4) config.hidden_size *= weight_shape[3];
            config.inter_size = weight_shape[1];
            config.group_size = (weight_shape.size() == 3) ? config.hidden_size : scale_shape[3];
            config.top_k = topk_shape.rbegin()->get_length();
            config.out_type = ov::element::dynamic;
            config.has_batch_dim = is_pa ? 0 : 1;

            args.push_back(pattern_map.at(input_gemm2_m));
            args.push_back(topk_weight_softmax);
            args.push_back(pattern_map.at(topk_indices_gemm2_m));
            // params for up
            args.push_back(pattern_map.at(compressed_weights_m_up));
            args.push_back(pattern_map.at(mul_const_m_up));
            if (pattern_map.count(sub_const_m_up)) {
                auto zp_shape = pattern_map.at(sub_const_m_up).get_node_shared_ptr()->get_shape();
                auto zp = std::make_shared<ov::op::v0::Convert>(pattern_map.at(sub_const_m_up), element::f16);
                args.push_back(zp);
                config.has_zp = true;
            }
            args.push_back(pattern_map.at(bias_up_gemm2_m));
            // params for down
            args.push_back(pattern_map.at(compressed_weights_m_down));
            args.push_back(pattern_map.at(mul_const_m_down));
            if (pattern_map.count(sub_const_m_down)) {
                auto zp = std::make_shared<ov::op::v0::Convert>(pattern_map.at(sub_const_m_down), element::f16);
                args.push_back(zp);
                config.has_zp = true;
            } else if (config.has_zp) {
                OPENVINO_THROW("gemm_down has no zp while gemm_up has zp!");
            }
            args.push_back(pattern_map.at(bias_down_gemm2_m));
            auto moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(args, config);
            moe_compressed->set_friendly_name(moe->get_friendly_name());
            ov::copy_runtime_info(moe, moe_compressed);
            ov::replace_node(moe, moe_compressed);
        } else {
            OPENVINO_THROW("Unsupported MOE expert type in ConvertMOEToMOECompressed");
        }
        std::cout << "ConvertMOEToMOECompressed|Successfully" << std::endl;
        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_root, "ConvertMOEToMOECompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu