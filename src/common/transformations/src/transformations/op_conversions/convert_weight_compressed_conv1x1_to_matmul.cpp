// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/op_conversions/convert_weight_compressed_conv1x1_to_matmul.hpp"

#include <iostream>
#include <ostream>
#include <vector>

#include "itt.hpp"
#include "openvino/core/graph_util.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/rt_info.hpp"
#include "openvino/core/rt_info/weightless_caching_attributes.hpp"
#include "openvino/core/type/element_type.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

ov::pass::ConvertWeightCompressedConv1x1ToMatmul::ConvertWeightCompressedConv1x1ToMatmul() {
    MATCHER_SCOPE(ConvertWeightCompressedConv1x1ToMatmul);

    auto first_input_m = ov::pass::pattern::any_input();
    auto a_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_activations_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({first_input_m, a_order_m});
    auto reshape_activations_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({first_input_m, a_order_m},
                                                          pattern::shape_matches("[?, hidden_in, 1, 1]"));
    auto direct_activation_m = ov::pass::pattern::any_input();
    auto a_m = std::make_shared<ov::pass::pattern::op::Or>(
        OutputVector{transpose_activations_m, reshape_activations_m, direct_activation_m});

    auto weights_const_m = wrap_type<ov::op::v0::Constant>(pattern::shape_matches("[?, ?, 1, 1]") ||
                                                           pattern::shape_matches("[?, ?, ?, 1, 1]"));
    auto weights_param_m = wrap_type<ov::op::v0::Parameter>(pattern::shape_matches("[?, ?, 1, 1]") ||
                                                            pattern::shape_matches("[?, ?, ?, 1, 1]"));
    auto weights_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{weights_const_m, weights_param_m});
    auto weight_convert_m = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({weights_m});
    auto weights_scales_m = ov::pass::pattern::any_input();
    auto weights_zp_m = ov::pass::pattern::any_input();
    auto weights_zp_convert_m = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({weights_zp_m});
    auto weight_subtract_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Subtract>({weight_convert_m, weights_zp_convert_m});
    // Make zp subtraction optional to account for symmetrical quantization cases
    auto weight_dequantized_m =
        std::make_shared<ov::pass::pattern::op::Or>(OutputVector{weight_convert_m, weight_subtract_m});
    auto weight_mult_m = ov::pass::pattern::wrap_type<ov::op::v1::Multiply>({weight_dequantized_m, weights_scales_m});
    auto weight_reshape_m =
        ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({weight_mult_m, ov::pass::pattern::any_input()});
    auto weight_input_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{weight_mult_m, weight_reshape_m});
    auto conv1x1_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>({a_m, weight_input_m});

    ov::matcher_pass_callback callback = [OV_CAPTURE_CPY_AND_THIS](Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();

        auto conv1x1 = ov::as_type_ptr<ov::op::v1::Convolution>(pattern_map.at(conv1x1_m).get_node_shared_ptr());
        auto weight_convert =
            ov::as_type_ptr<ov::op::v0::Convert>(pattern_map.at(weight_convert_m).get_node_shared_ptr());
        auto weight_sub = (pattern_map.count(weight_subtract_m) > 0)
                              ? pattern_map.at(weight_subtract_m).get_node_shared_ptr()
                              : nullptr;
        auto weight_mult = ov::as_type_ptr<ov::op::v1::Multiply>(pattern_map.at(weight_mult_m).get_node_shared_ptr());
        auto weight_reshape = (pattern_map.count(weight_reshape_m) > 0)
                                  ? pattern_map.at(weight_reshape_m).get_node_shared_ptr()
                                  : nullptr;
        if (!conv1x1 || transformation_callback(conv1x1)) {
            return false;
        }

        // Dynamically detect optional bias, convert, and output transpose/reshape
        // by walking consumers forward from conv1x1.
        auto get_single_consumer = [](const std::shared_ptr<Node>& node) -> std::shared_ptr<Node> {
            auto consumers = node->get_output_target_inputs(0);
            return (consumers.size() == 1) ? consumers.begin()->get_node()->shared_from_this() : nullptr;
        };

        std::shared_ptr<Node> outermost = conv1x1;
        std::shared_ptr<Node> bias_add = nullptr;
        std::shared_ptr<ov::op::v0::Constant> bias_const = nullptr;
        std::shared_ptr<Node> convert_out = nullptr;

        // Walk through optional bias (Add with constant shaped [1, ?, 1, 1])
        if (auto consumer = get_single_consumer(outermost)) {
            if (ov::is_type<ov::op::v1::Add>(consumer)) {
                for (size_t i = 0; i < consumer->get_input_size(); i++) {
                    if (auto c = ov::as_type_ptr<ov::op::v0::Constant>(consumer->get_input_node_shared_ptr(i))) {
                        auto s = c->get_shape();
                        if (s.size() == 4 && s[0] == 1 && s[2] == 1 && s[3] == 1) {
                            bias_add = consumer;
                            bias_const = c;
                            outermost = consumer;
                            break;
                        }
                    }
                }
            }
        }
        // Walk through optional Convert
        if (auto consumer = get_single_consumer(outermost)) {
            if (ov::is_type<ov::op::v0::Convert>(consumer)) {
                convert_out = consumer;
                outermost = consumer;
            }
        }
        // Detect output transpose or reshape consumer
        std::shared_ptr<ov::Node> consumer_transpose = nullptr;
        std::shared_ptr<ov::Node> consumer_reshape = nullptr;
        if (auto consumer = get_single_consumer(outermost)) {
            if (ov::is_type<ov::op::v1::Transpose>(consumer)) {
                consumer_transpose = consumer;
            } else if (ov::is_type<ov::op::v1::Reshape>(consumer)) {
                consumer_reshape = consumer;
            }
        }

        auto weight = pattern_map.at(weights_m).get_node_shared_ptr();
        auto scale = pattern_map.at(weights_scales_m).get_node_shared_ptr();
        auto zp = (pattern_map.count(weights_zp_m) > 0) ? pattern_map.at(weights_zp_m).get_node_shared_ptr() : nullptr;

        // Determine activation source based on which pattern branch matched
        std::shared_ptr<Node> activation;
        bool has_input_transpose = pattern_map.count(transpose_activations_m) > 0;
        bool has_input_reshape = pattern_map.count(reshape_activations_m) > 0;
        if (has_input_transpose || has_input_reshape) {
            activation = pattern_map.at(first_input_m).get_node_shared_ptr();
        } else {
            // Direct activation (no transpose/reshape) - use the conv's input directly
            activation = pattern_map.at(direct_activation_m).get_node_shared_ptr();
        }

        auto reshape_const_to_2d = [](std::shared_ptr<ov::Node> node) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            OPENVINO_ASSERT(constant != nullptr);
            ov::Shape current_shape = constant->get_shape();
            if (current_shape.size() == 2)
                return constant;

            if (current_shape.size() <= 1) {
                auto new_shape = ov::Shape{(current_shape.size() == 1) ? current_shape[0] : 1, 1};

                auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, new_shape);

                ov::copy_weightless_cache_attr(constant, new_constant);
                return new_constant;
            } else if (current_shape.size() == 4) {
                OPENVINO_ASSERT(current_shape[2] == 1 && current_shape[3] == 1);

                auto new_shape = ov::Shape{current_shape[0], current_shape[1]};

                auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, new_shape);

                ov::copy_weightless_cache_attr(constant, new_constant);
                return new_constant;
            } else {
                OPENVINO_ASSERT(current_shape.size() == 5);
                OPENVINO_ASSERT(current_shape[3] == 1 && current_shape[4] == 1);

                auto new_shape = ov::Shape{current_shape[0], current_shape[1], current_shape[2]};

                auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, new_shape);

                ov::copy_weightless_cache_attr(constant, new_constant);
                return new_constant;
            }
        };

        // add reshape after weight
        std::shared_ptr<ov::op::v0::Convert> weight_squeezed_convert;
        if (ov::as_type_ptr<ov::op::v0::Constant>(weight)) {
            auto Reshape_weight = reshape_const_to_2d(weight);
            MatcherPass::register_new_node(Reshape_weight);
            Reshape_weight->set_friendly_name(weight->get_friendly_name() + "_Reshape_weight");
            weight_squeezed_convert =
                ov::as_type_ptr<ov::op::v0::Convert>(weight_convert->clone_with_new_inputs({Reshape_weight}));
            ov::copy_runtime_info(weight_convert, weight_squeezed_convert);
        } else {
            auto param = ov::as_type_ptr<ov::op::v0::Parameter>(weight);
            OPENVINO_ASSERT(param != nullptr);
            std::vector<int> values_reshape_b;
            auto shape_b = param->get_output_partial_shape(0);
            // removed last two dimensions, 1x1
            for (size_t i = 0; i < (shape_b.size() - 2); i++)
                values_reshape_b.push_back(static_cast<int>(shape_b.to_shape()[i]));

            auto reshape_weight_const =
                ov::op::v0::Constant::create(element::i32, Shape{values_reshape_b.size()}, values_reshape_b);
            auto Reshape_weight = std::make_shared<ov::op::v1::Reshape>(param, reshape_weight_const, false);
            MatcherPass::register_new_node(Reshape_weight);
            Reshape_weight->set_friendly_name(param->get_friendly_name() + "_Reshape_weight");
            weight_squeezed_convert =
                ov::as_type_ptr<ov::op::v0::Convert>(weight_convert->clone_with_new_inputs({Reshape_weight}));
            ov::copy_runtime_info(weight_convert, {weight_squeezed_convert, Reshape_weight});
        }
        ov::disable_constant_folding(weight_squeezed_convert);

        // add reshape after scales
        auto Reshape_scale = reshape_const_to_2d(scale);
        MatcherPass::register_new_node(Reshape_scale);
        Reshape_scale->set_friendly_name(scale->get_friendly_name() + "_Reshape_scale");
        ov::copy_runtime_info(scale, Reshape_scale);

        auto scaled_weight = weight_mult->clone_with_new_inputs({weight_squeezed_convert, Reshape_scale});
        if (zp) {
            // add reshape after zero points
            auto Reshape_zp = reshape_const_to_2d(zp);
            MatcherPass::register_new_node(Reshape_zp);
            Reshape_zp->set_friendly_name(zp->get_friendly_name() + "_Reshape_zp");
            auto weights_zp_convert =
                ov::as_type_ptr<ov::op::v0::Convert>(pattern_map.at(weights_zp_convert_m).get_node_shared_ptr());
            auto zp_squeezed_convert = weights_zp_convert->clone_with_new_inputs({Reshape_zp});
            ov::copy_runtime_info(weights_zp_convert, zp_squeezed_convert);
            ov::disable_constant_folding(zp_squeezed_convert);
            auto zero_adjusted_weight =
                weight_sub->clone_with_new_inputs({weight_squeezed_convert, zp_squeezed_convert});
            ov::copy_runtime_info(weight_sub, zero_adjusted_weight);
            scaled_weight = weight_mult->clone_with_new_inputs({zero_adjusted_weight, Reshape_scale});
        }
        ov::copy_runtime_info(weight_mult, scaled_weight);
        ov::disable_constant_folding(scaled_weight);

        if (weight_reshape) {
            const auto& shape = scaled_weight->get_output_partial_shape(0);
            OPENVINO_ASSERT(shape.rank().get_length() == 3, "Expected 3 Dim weights for block decompression case");
            auto shape_const = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64,
                ov::Shape{2},
                std::vector<int64_t>{shape[0].get_length(), shape[1].get_length() * shape[2].get_length()});
            auto final_weight_reshape = std::make_shared<ov::op::v1::Reshape>(scaled_weight, shape_const, false);
            ov::copy_runtime_info(weight_reshape, final_weight_reshape);
            final_weight_reshape->set_friendly_name(weight_reshape->get_friendly_name() + "_reshape_weight");
            scaled_weight = final_weight_reshape;
        }

        // When activation is reshaped to [?, hidden_in, 1, 1], two possible cases:
        // 1. reshape from [..., hidden_in]
        //    direct use it in matmul.
        // 2. reshape from [..., num_head, head_dim]
        //    can't use it directly, need reshape it to [..., hidden_in], then use in matmul.
        if (has_input_reshape) {
            auto reshape_activations = pattern_map.at(reshape_activations_m).get_node_shared_ptr();
            auto shape_in = reshape_activations->get_input_partial_shape(0);
            auto shape_out = reshape_activations->get_output_partial_shape(0);
            if (shape_in[-1].is_dynamic() || shape_in[-1].get_length() != shape_out[1].get_length()) {
                auto reshape_const =
                    std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                           ov::Shape{4},
                                                           std::vector<int64_t>{1, 1, -1, shape_out[1].get_length()});
                auto reshape_activations_new = std::make_shared<ov::op::v1::Reshape>(activation, reshape_const, false);
                ov::copy_runtime_info(reshape_activations, reshape_activations_new);
                activation = reshape_activations_new;
            }
        } else if (!has_input_transpose && !has_input_reshape) {
            // No transpose or reshape on activation input - activation is in NCHW [N, C, H, W].
            // Insert NCHW->NHWC transpose [0, 2, 3, 1] so matmul gets [..., C] as last dim.
            auto nchw_to_nhwc_order = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                             ov::Shape{4},
                                                                             std::vector<int64_t>{0, 2, 3, 1});
            auto transpose_input = std::make_shared<ov::op::v1::Transpose>(activation, nchw_to_nhwc_order);
            transpose_input->set_friendly_name(activation->get_friendly_name() + "_nchw_to_nhwc");
            ov::copy_runtime_info(conv1x1, transpose_input);
            activation = transpose_input;
        }

        // If the activation has a static leading dimension of 1, squeeze it.
        // This is done to allow pre-selection of OCL implementations for non-IMMAD devices, reducing memory pressure.
        bool squeeze_activation = false;
        auto act_pshape = activation->get_output_partial_shape(0);
        if (act_pshape.rank().is_static() && act_pshape.rank().get_length() >= 4 && act_pshape[0].is_static() &&
            act_pshape[0] == 1) {
            squeeze_activation = true;
            auto squeeze_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{3},
                                                       std::vector<int64_t>{1, -1, act_pshape[-1].get_length()});
            auto squeeze = std::make_shared<ov::op::v1::Reshape>(activation, squeeze_const, false);
            ov::copy_runtime_info(activation, squeeze);
            squeeze->set_friendly_name(activation->get_friendly_name() + "_squeeze");
            activation = squeeze;
        }

        auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, scaled_weight, false, true);
        ov::copy_runtime_info(conv1x1, matmul);
        std::shared_ptr<Node> matmul_out;
        if (bias_add) {
            auto bias = bias_const;
            OPENVINO_ASSERT(bias != nullptr);

            ov::Shape bias_shape = bias->get_shape();
            OPENVINO_ASSERT(bias_shape.size() == 4);

            auto new_bias_shape = ov::Shape{bias_shape[0], bias_shape[2], bias_shape[3], bias_shape[1]};

            auto Reshape_bias = std::make_shared<ov::op::v0::Constant>(*bias, new_bias_shape);
            ov::copy_runtime_info(bias, Reshape_bias);

            ov::copy_weightless_cache_attr(bias, Reshape_bias);
            MatcherPass::register_new_node(Reshape_bias);
            Reshape_bias->set_friendly_name(bias->get_friendly_name() + "_Reshape_bias");

            matmul_out = bias_add->clone_with_new_inputs({matmul, Reshape_bias});
            ov::copy_runtime_info(bias_add, matmul_out);
        } else {
            matmul_out = matmul;
        }

        if (squeeze_activation) {
            auto shape_out = matmul_out->get_output_partial_shape(0);
            auto unsqueeze_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{4},
                                                       std::vector<int64_t>{1, 1, -1, shape_out[-1].get_length()});
            auto unsqueeze = std::make_shared<ov::op::v1::Reshape>(matmul_out, unsqueeze_const, false);
            ov::copy_runtime_info(matmul_out, unsqueeze);
            unsqueeze->set_friendly_name(matmul_out->get_friendly_name() + "_unsqueeze");
            matmul_out = unsqueeze;
        }

        // Build final output, optionally wrapping in Convert
        std::shared_ptr<Node> final_out = matmul_out;
        if (convert_out) {
            auto convert_final = convert_out->clone_with_new_inputs({matmul_out});
            ov::copy_runtime_info(convert_out, convert_final);
            final_out = convert_final;
        }

        if (consumer_transpose) {
            // Transpose consumer found - absorb it, matmul replaces conv+transpose
            final_out->set_friendly_name(consumer_transpose->get_friendly_name());
            ov::copy_runtime_info(m.get_matched_nodes(), final_out);
            ov::replace_node(consumer_transpose, final_out);
        } else if (consumer_reshape) {
            // Reshape consumer found - clone reshape with matmul output
            auto reshape_order = consumer_reshape->get_input_node_shared_ptr(1);
            auto reshape_final = consumer_reshape->clone_with_new_inputs({final_out, reshape_order});
            reshape_final->set_friendly_name(consumer_reshape->get_friendly_name());
            ov::copy_runtime_info(m.get_matched_nodes(), reshape_final);
            ov::replace_node(consumer_reshape, reshape_final);
        } else {
            // No transpose or reshape consumer - add NHWC->NCHW transpose
            auto nhwc_to_nchw_order = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                             ov::Shape{4},
                                                                             std::vector<int64_t>{0, 3, 1, 2});
            auto transpose_to_nchw = std::make_shared<ov::op::v1::Transpose>(final_out, nhwc_to_nchw_order);
            transpose_to_nchw->set_friendly_name(outermost->get_friendly_name());
            ov::copy_runtime_info(m.get_matched_nodes(), transpose_to_nchw);
            ov::replace_node(outermost, transpose_to_nchw);
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(conv1x1_m, matcher_name);
    this->register_matcher(m, callback);
}
