// Copyright (C) 2018-2026 Intel Corporation
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
#include "openvino/op/swish.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/moe.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/or.hpp"
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
    auto gemm3_compressed_weights_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches_any({ov::element::u4,ov::element::u8}));\
    auto gemm3_zp_m_##SUFFIX = wrap_type<ov::op::v0::Constant>(type_matches_any({ov::element::u4, ov::element::u8}));\
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
        return in_ps.rank().is_static() && out_ps.rank().is_static() &&\
        ((in_ps.size() == 4 && out_ps.size() == 3) || (in_ps.size() == 3 && out_ps.size() == 2));\
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

     // shared expert pattern
    MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(shared_gate);
    MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(shared_up);
    MOE_COMPRESSED_WEIGHT_GEMM3_PATTERN(shared_down);

    auto hidden_states_m = any_input();
    auto routing_weights_m = any_input();
    auto topk_m = any_input();

    auto moe_root_gemm3_no_shared_expert_bare =
        wrap_type<ov::op::internal::MOE>({hidden_states_m, routing_weights_m, topk_m, gemm3_convert_m_gate, gemm3_convert_m_up, gemm3_convert_m_down},
                                         [](const ov::Output<ov::Node>& output) {
                                             auto moe = ov::as_type_ptr<ov::op::internal::MOE>(output.get_node_shared_ptr());
                                             return moe && moe->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
                                         });

    auto moe_root_gemm3_no_shared_expert =
        wrap_type<ov::op::internal::MOE>({hidden_states_m, routing_weights_m, topk_m, gemm3_convert_m_gate, gemm3_convert_m_up, gemm3_convert_m_down},
                                         [](const ov::Output<ov::Node>& output) {
                                             auto moe = ov::as_type_ptr<ov::op::internal::MOE>(output.get_node_shared_ptr());
                                             if (!moe || moe->get_config().expert_type != ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU) return false;
                                             for (auto& tg : output.get_target_inputs()) {
                                                 if (ov::is_type<ov::op::v1::Add>(tg.get_node())) {
                                                     auto add = tg.get_node();
                                                     // Check both Add inputs; follow through Reshape to detect shared expert
                                                     for (size_t i = 0; i < add->get_input_size(); i++) {
                                                         auto n = add->get_input_node_ptr(i);
                                                         if (n && ov::is_type<ov::op::v1::Reshape>(n))
                                                             n = n->get_input_node_ptr(0);
                                                         if (n && ov::is_type<ov::op::v1::Multiply>(n)) return false;
                                                     }
                                                 }
                                             }
                                             return true;
                                         });

    // Shared expert uses a separate hidden_states input because in the actual model,
    // MOE's hidden_states is the node BEFORE a Reshape (matmul_experts_fusion extracts input_value(0)),
    // while shared expert MatMuls take the node AFTER that Reshape.
    auto shared_hidden_states_m = any_input();
    auto shared_gate_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_states_m, gemm3_convert_m_shared_gate});
    auto shared_swish_m = wrap_type<ov::op::v4::Swish>({shared_gate_m});
    auto shared_up_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_states_m, gemm3_convert_m_shared_up});
    auto shared_mul_m = wrap_type<ov::op::v1::Multiply>({shared_swish_m, shared_up_m});
    auto shared_down_m = wrap_type<ov::op::v0::MatMul>({shared_mul_m, gemm3_convert_m_shared_down});

    auto shared_gate_gate_wei_m = any_input();
    auto shared_gate_gate_m = wrap_type<ov::op::v0::MatMul>({shared_hidden_states_m, shared_gate_gate_wei_m});
    auto shared_gate_sigmoid_m = wrap_type<ov::op::v0::Sigmoid>({shared_gate_gate_m});
    auto shared_expert_gated_m = wrap_type<ov::op::v1::Multiply>({shared_gate_sigmoid_m, shared_down_m});
    auto shared_expert_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{shared_down_m, shared_expert_gated_m});
    auto shared_expert_reshaped_m = optional<ov::op::v1::Reshape>({shared_expert_m, any_input()});

    auto add_1 = wrap_type<ov::op::v1::Add>({moe_root_gemm3_no_shared_expert_bare, shared_expert_reshaped_m});
    auto add_2 = wrap_type<ov::op::v1::Add>({shared_expert_reshaped_m, moe_root_gemm3_no_shared_expert_bare});
    auto moe_root_gemm3_shared_expert = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{add_1, add_2});
    auto moe_root_gemm3 = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{moe_root_gemm3_no_shared_expert, moe_root_gemm3_shared_expert});
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
                                             return moe_op &&
                                                    moe_op->get_config().expert_type == ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP;
                                         });

    // gemm2 pattern finished

    auto moe_root = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{moe_root_gemm3, moe_root_gemm2});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto root_node = pattern_map.at(moe_root).get_node_shared_ptr();
        std::shared_ptr<ov::op::internal::MOE> moe;
        if (auto add_op = ov::as_type_ptr<ov::op::v1::Add>(root_node)) {
            moe = ov::as_type_ptr<ov::op::internal::MOE>(pattern_map.count(moe_root_gemm3_no_shared_expert_bare) ? pattern_map.at(moe_root_gemm3_no_shared_expert_bare).get_node_shared_ptr() : pattern_map.at(moe_root_gemm3_no_shared_expert).get_node_shared_ptr());
        } else {
            moe = ov::as_type_ptr<ov::op::internal::MOE>(root_node);
        }
        if (!moe || transformation_callback(root_node)) {
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
            bool has_shared_expert = pattern_map.count(gemm3_compressed_weights_m_shared_gate) > 0;

            OutputVector args(has_shared_expert ? 22 : 12);
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

            if (has_shared_expert) {
                args[12] = pattern_map.at(gemm3_compressed_weights_m_shared_gate);
                if (group_compressed) {
                    MOE_COMPRESSED_WEIGHT_GEMM3(shared_gate);
                    args[13] = gemm3_transpose_scale_shared_gate;
                    args[14] = gemm3_transpose_zp_shared_gate;
                } else {
                    args[13] = pattern_map.at(gemm3_scale_m_shared_gate);
                    args[14] = pattern_map.at(gemm3_zp_m_shared_gate);
                }
                args[15] = pattern_map.at(gemm3_compressed_weights_m_shared_up);
                if (group_compressed) {
                    MOE_COMPRESSED_WEIGHT_GEMM3(shared_up);
                    args[16] = gemm3_transpose_scale_shared_up;
                    args[17] = gemm3_transpose_zp_shared_up;
                } else {
                    args[16] = pattern_map.at(gemm3_scale_m_shared_up);
                    args[17] = pattern_map.at(gemm3_zp_m_shared_up);
                }
                args[18] = pattern_map.at(gemm3_compressed_weights_m_shared_down);
                if (group_compressed) {
                    MOE_COMPRESSED_WEIGHT_GEMM3(shared_down);
                    args[19] = gemm3_transpose_scale_shared_down;
                    args[20] = gemm3_transpose_zp_shared_down;
                } else {
                    args[19] = pattern_map.at(gemm3_scale_m_shared_down);
                    args[20] = pattern_map.at(gemm3_zp_m_shared_down);
                }
                if (pattern_map.count(shared_gate_gate_wei_m)) {
                    args[21] = pattern_map.at(shared_gate_gate_wei_m);
                } else {
                    size_t hidden_size = group_compressed ? weight_shape[2] * weight_shape[3] : weight_shape[2];
                    args[21] = ov::op::v0::Constant::create(ov::element::f16, ov::Shape{hidden_size}, std::vector<float>(hidden_size, 0.0f));
                }
            }
            ov::intel_gpu::op::MOECompressed::Config config(moe->get_config());
            config.hidden_size = group_compressed ? weight_shape[2] * weight_shape[3] : weight_shape[2];
            config.inter_size = weight_shape[1];
            config.num_expert = weight_shape[0];
            config.num_shared_expert = has_shared_expert ? 1 : 0;
            config.group_size = group_compressed ? weight_shape[3] : std::numeric_limits<size_t>::max();
            auto topk_shape = pattern_map.at(topk_m).get_partial_shape();
            if (!topk_shape[1].is_static()) {
                OPENVINO_THROW("K dimenion in moe topk input should be static..");
            }
            config.top_k = topk_shape[1].get_length();
            config.out_type = ov::element::f16;
            config.has_batch_dim = is_pa ? 0 : 1;
            std::shared_ptr<ov::Node> moe_compressed = std::make_shared<ov::intel_gpu::op::MOECompressed>(args, config);
            moe_compressed->set_friendly_name(moe->get_friendly_name());
            ov::copy_runtime_info(moe, moe_compressed);

            // Since f16 precision is forced for MOECompressed output, we may need to insert Convert after MOECompressed
            // in order not to break the model semantic. This Convert will be most likely optimized at ConvertPrecision stage
            if (moe->get_output_element_type(0) != moe_compressed->get_output_element_type(0)) {
                moe_compressed->set_friendly_name(moe_compressed->get_friendly_name() + "/MOECompressed");
                moe_compressed = std::make_shared<ov::op::v0::Convert>(moe_compressed, moe->get_output_element_type(0));
                moe_compressed->set_friendly_name(moe->get_friendly_name());
                ov::copy_runtime_info(moe, moe_compressed);
            }
            if (has_shared_expert) {
                moe_compressed->set_friendly_name(root_node->get_friendly_name());
                ov::copy_runtime_info(root_node, moe_compressed);
                ov::replace_node(root_node, moe_compressed);
            } else {
                ov::replace_node(moe, moe_compressed);
            }
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
            moe_compressed->set_friendly_name(root_node->get_friendly_name());
            ov::copy_runtime_info(root_node, moe_compressed);
            ov::replace_node(root_node, moe_compressed);
        } else {
            OPENVINO_THROW("Unsupported MOE expert type in ConvertMOEToMOECompressed");
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(moe_root, "ConvertMOEToMOECompressed");
    this->register_matcher(m, callback);
}

}  // namespace ov::intel_gpu