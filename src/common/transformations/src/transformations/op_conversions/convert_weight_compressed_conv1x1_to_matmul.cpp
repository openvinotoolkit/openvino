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
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/pattern/op/any.hpp"
#include "openvino/pass/pattern/op/label.hpp"
#include "openvino/pass/pattern/op/optional.hpp"
#include "openvino/pass/pattern/op/or.hpp"
#include "openvino/pass/pattern/op/pattern.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"
#include "transformations/utils/utils.hpp"

using namespace ov::pass::pattern;
using ov::pass::pattern::op::Or;

ov::pass::ConvertWeightCompressedConv1x1ToMatmul::ConvertWeightCompressedConv1x1ToMatmul() {
    MATCHER_SCOPE(ConvertWeightCompressedConv1x1ToMatmul);
    auto filter1x1_path = [](const ov::Output<ov::Node>& output) {
        const auto& pshape = output.get_partial_shape();
        return ov::op::util::is_on_path<ov::op::v0::Constant, ov::op::v0::Parameter>(output) && pshape.is_static() &&
               pshape[-1] == 1 && pshape[-2] == 1;
    };

    auto bias_path = [](const ov::Output<ov::Node>& output) {
        const auto& pshape = output.get_partial_shape();
        return ov::op::util::is_on_path<ov::op::v0::Constant>(output) && pshape.is_static() && pshape[0] == 1 &&
               pshape[2] == 1 && pshape[3] == 1;
    };

    auto first_input_m = ov::pass::pattern::any_input();
    auto a_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_activations_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({first_input_m, a_order_m});
    auto reshape_activations_m = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({first_input_m, a_order_m});
    auto a_m =
        std::make_shared<ov::pass::pattern::op::Or>(OutputVector{transpose_activations_m, reshape_activations_m});

    auto weights_const_m =
        wrap_type<ov::op::v0::Constant>((rank_equals(4) || rank_equals(5)) && has_static_rank() && filter1x1_path);
    auto weights_param_m =
        wrap_type<ov::op::v0::Parameter>((rank_equals(4) || rank_equals(5)) && has_static_rank() && filter1x1_path);
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

    // Optional bias
    auto bias_const_m = wrap_type<ov::op::v0::Constant>(rank_equals(4) && has_static_rank() && bias_path);
    auto bias_m = ov::pass::pattern::wrap_type<ov::op::v1::Add>({conv1x1_m, bias_const_m});
    auto bias_out_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{conv1x1_m, bias_m});

    auto convert_m = ov::pass::pattern::wrap_type<ov::op::v0::Convert>({bias_out_m});
    auto conv_out_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{bias_out_m, convert_m});

    auto c_order_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto transpose_output_m = ov::pass::pattern::wrap_type<ov::op::v1::Transpose>({conv_out_m, c_order_m});
    auto reshape_output_m = ov::pass::pattern::wrap_type<ov::op::v1::Reshape>({conv_out_m, c_order_m});
    auto output_m = std::make_shared<ov::pass::pattern::op::Or>(OutputVector{transpose_output_m, reshape_output_m});

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
        auto bias_out = (pattern_map.count(bias_m) > 0) ? pattern_map.at(bias_m).get_node_shared_ptr() : nullptr;
        auto bias_const =
            (pattern_map.count(bias_const_m) > 0) ? pattern_map.at(bias_const_m).get_node_shared_ptr() : nullptr;
        auto convert_out =
            (pattern_map.count(convert_m) > 0) ? pattern_map.at(convert_m).get_node_shared_ptr() : nullptr;
        auto out_order = (pattern_map.count(c_order_m) > 0) ? pattern_map.at(c_order_m).get_node_shared_ptr() : nullptr;
        auto reshape_out = (pattern_map.count(reshape_output_m) > 0)
                               ? pattern_map.at(reshape_output_m).get_node_shared_ptr()
                               : nullptr;
        if (!conv1x1 || transformation_callback(conv1x1)) {
            return false;
        }

        auto weight = pattern_map.at(weights_m).get_node_shared_ptr();
        auto scale = pattern_map.at(weights_scales_m).get_node_shared_ptr();
        auto zp = (pattern_map.count(weights_zp_m) > 0) ? pattern_map.at(weights_zp_m).get_node_shared_ptr() : nullptr;
        auto activation = pattern_map.at(first_input_m).get_node_shared_ptr();

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
            for (size_t i = 0; i < shape_b.size(); i++)
                if (shape_b.to_shape()[i] != 1) {
                    values_reshape_b.push_back(static_cast<int>(shape_b.to_shape()[i]));
                }

            auto reshape_weight_const =
                ov::op::v0::Constant::create(element::i32, Shape{values_reshape_b.size()}, values_reshape_b);
            auto Reshape_weight = std::make_shared<ov::op::v1::Reshape>(param, reshape_weight_const, false);
            MatcherPass::register_new_node(Reshape_weight);
            Reshape_weight->set_friendly_name(param->get_friendly_name() + "_Reshape_weight");
            weight_squeezed_convert =
                ov::as_type_ptr<ov::op::v0::Convert>(weight_convert->clone_with_new_inputs({Reshape_weight}));
            ov::copy_runtime_info(weight_convert, weight_squeezed_convert);
            ov::copy_runtime_info(weight_convert, Reshape_weight);
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
            OPENVINO_ASSERT(shape.rank().get_length() == 3, "Expected 3 Dim weights for block quantization case");
            auto shape_const = std::make_shared<ov::op::v0::Constant>(
                ov::element::i64,
                ov::Shape{2},
                std::vector<int64_t>{shape[0].get_length(), shape[1].get_length() * shape[2].get_length()});
            auto final_weight_reshape = std::make_shared<ov::op::v1::Reshape>(scaled_weight, shape_const, false);
            ov::copy_runtime_info(weight_reshape, final_weight_reshape);
            final_weight_reshape->set_friendly_name(weight_reshape->get_friendly_name() + "_reshape_weight");
            scaled_weight = final_weight_reshape;
        }

        auto matmul = std::make_shared<ov::op::v0::MatMul>(activation, scaled_weight, false, true);
        ov::copy_runtime_info(conv1x1, matmul);
        std::shared_ptr<Node> matmul_out;
        if (bias_out) {
            auto bias = ov::as_type_ptr<ov::op::v0::Constant>(bias_const);
            OPENVINO_ASSERT(bias != nullptr);

            ov::Shape bias_shape = bias->get_shape();
            OPENVINO_ASSERT(bias_shape.size() == 4);

            auto new_bias_shape = ov::Shape{bias_shape[0], bias_shape[2], bias_shape[3], bias_shape[1]};

            auto Reshape_bias = std::make_shared<ov::op::v0::Constant>(*bias, new_bias_shape);
            ov::copy_runtime_info(bias, Reshape_bias);

            ov::copy_weightless_cache_attr(bias, Reshape_bias);
            MatcherPass::register_new_node(Reshape_bias);
            Reshape_bias->set_friendly_name(bias->get_friendly_name() + "_Reshape_bias");

            matmul_out = bias_out->clone_with_new_inputs({matmul, Reshape_bias});
            ov::copy_runtime_info(bias_out, matmul_out);
        } else {
            matmul_out = matmul;
        }

        if (reshape_out) {
            if (convert_out) {
                auto convert_final = convert_out->clone_with_new_inputs({matmul_out});
                auto reshape_final = reshape_out->clone_with_new_inputs({convert_final, out_order});
                reshape_final->set_friendly_name(m.get_match_root()->get_friendly_name());
                ov::copy_runtime_info(convert_out, convert_final);
                ov::copy_runtime_info(m.get_matched_nodes(), reshape_final);
                ov::replace_node(m.get_match_root(), reshape_final);
            } else {
                auto reshape_final = reshape_out->clone_with_new_inputs({matmul_out, out_order});
                reshape_final->set_friendly_name(m.get_match_root()->get_friendly_name());
                ov::copy_runtime_info(m.get_matched_nodes(), reshape_final);
                ov::replace_node(m.get_match_root(), reshape_final);
            }
        } else {
            if (convert_out) {
                auto convert_final = convert_out->clone_with_new_inputs({matmul_out});
                convert_final->set_friendly_name(m.get_match_root()->get_friendly_name());
                ov::copy_runtime_info(m.get_matched_nodes(), convert_final);
                ov::replace_node(m.get_match_root(), convert_final);
            } else {
                matmul_out->set_friendly_name(m.get_match_root()->get_friendly_name());
                ov::copy_runtime_info(m.get_matched_nodes(), matmul_out);
                ov::replace_node(m.get_match_root(), matmul_out);
            }
        }

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(output_m, matcher_name);
    this->register_matcher(m, callback);
}

ov::pass::ConvertWeightCompressedConv1x1ToMatmul_ActNotTran::ConvertWeightCompressedConv1x1ToMatmul_ActNotTran(
    const element::TypeVector& supported_precisions,
    const element::TypeVector& unsupported_precisions) {
    MATCHER_SCOPE(ConvertWeightCompressedConv1x1ToMatmul_ActNotTran);

    auto final_precisions = supported_precisions;
    if (!unsupported_precisions.empty()) {
        final_precisions.erase(std::remove_if(final_precisions.begin(),
                                              final_precisions.end(),
                                              [&](const ov::element::Type& type) {
                                                  return std::find(unsupported_precisions.begin(),
                                                                   unsupported_precisions.end(),
                                                                   type) != unsupported_precisions.end();
                                              }),
                               final_precisions.end());
    }

    // compressed weight, ranks are [hidden_out, block_num, block_size, 1, 1]
    auto weights_5d_m =
        pattern::wrap_type<ov::op::v0::Constant>(pattern::type_matches_any(final_precisions) &&
                                                 pattern::shape_matches("[hidden_out, block_num, block_size, 1, 1]"));
    auto weights_5d_convert_m = pattern::wrap_type<ov::op::v0::Convert>({weights_5d_m});
    auto weights_4d_m = pattern::wrap_type<ov::op::v0::Constant>(
        pattern::type_matches_any(final_precisions) && pattern::shape_matches("[hidden_out, hidden_in, 1, 1]"));
    auto weights_4d_reshape_m =
        pattern::wrap_type<ov::op::v1::Reshape>({weights_4d_m, pattern::any_input()},
                                                pattern::shape_matches("[hidden_out, block_num, block_size, 1, 1]"));
    auto weights_4d_convert_m = pattern::wrap_type<ov::op::v0::Convert>({weights_4d_reshape_m});
    auto weights_convert_m = weights_5d_convert_m | weights_4d_convert_m;

    // zero_point, ranks are [hidden_out, block_num, 1, 1, 1]
    auto zp_5d_m = pattern::wrap_type<ov::op::v0::Constant>(pattern::rank_equals(5) &&
                                                            pattern::shape_matches("[hidden_out, block_num, 1, 1, 1]"));
    auto zp_5d_convert_m = pattern::wrap_type<ov::op::v0::Convert>({zp_5d_m});
    auto zp_4d_m = pattern::wrap_type<ov::op::v0::Constant>(pattern::shape_matches("[hidden_out, block_num, 1, 1]"));
    auto zp_4d_unsqueeze_m =
        pattern::wrap_type<ov::op::v0::Unsqueeze>({zp_4d_m, pattern::any_input()},
                                                  pattern::shape_matches("[hidden_out, block_num, 1, 1, 1]"));
    auto zp_4d_convert_m = pattern::wrap_type<ov::op::v0::Convert>({zp_4d_unsqueeze_m});
    auto zp_convert_m = zp_5d_convert_m | zp_4d_convert_m;

    auto weights_sub_m = pattern::optional<ov::op::v1::Subtract>({weights_convert_m, zp_convert_m});

    // scale, ranks are [hidden_out, block_num, 1, 1, 1]
    auto scale_5d_m =
        pattern::wrap_type<ov::op::v0::Constant>(pattern::shape_matches("[hidden_out, block_num, 1, 1, 1]"));
    auto scale_4d_m = pattern::wrap_type<ov::op::v0::Constant>(pattern::shape_matches("[hidden_out, block_num, 1, 1]"));
    auto scale_4d_unsqueeze_m =
        pattern::wrap_type<ov::op::v0::Unsqueeze>({scale_4d_m, pattern::any_input()},
                                                  pattern::shape_matches("[hidden_out, block_num, 1, 1, 1]"));
    auto scale_m = scale_5d_m | scale_4d_unsqueeze_m;

    auto weights_mult_m = pattern::wrap_type<ov::op::v1::Multiply>({weights_sub_m, scale_m});

    // decompressed weights for convolution, reshape to [hidden_out, hidden_in, 1, 1]
    auto weights_reshape_m =
        pattern::wrap_type<ov::op::v1::Reshape>({weights_mult_m, pattern::any_input()},
                                                pattern::shape_matches("[hidden_out, hidden_in, 1, 1]"));

    auto conv_input_1 = pattern::any_input(pattern::shape_matches("[?, ?, 1, 1]"));
    auto conv_input_2 = pattern::any_input(pattern::shape_matches("[1, ?, 1, ?]"));
    auto conv_input_3 = pattern::any_input(pattern::shape_matches("[1, ?, ?, 1]"));
    auto conv_input_m = conv_input_1 | conv_input_2 | conv_input_3;

    auto conv_pattern_m = pattern::wrap_type<ov::op::v1::Convolution>({conv_input_m, weights_reshape_m},
                                                                      {{"auto_pad", "explicit"},
                                                                       {"dilations", std::vector<int64_t>{1, 1}},
                                                                       {"strides", std::vector<int64_t>{1, 1}},
                                                                       {"pads_begin", std::vector<int64_t>{0, 0}},
                                                                       {"pads_end", std::vector<int64_t>{0, 0}}});

    // Add bias
    auto bias_const_m = pattern::wrap_type<ov::op::v0::Constant>(pattern::shape_matches("[1, hidden_out, 1, 1]"));
    auto bias_input_m = pattern::any_input(pattern::shape_matches("[1, hidden_out, 1, 1]"));
    auto bias_m = bias_const_m | bias_input_m;
    auto bias_out_m = pattern::wrap_type<ov::op::v1::Add>({conv_pattern_m, bias_m});

    matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        auto conv_node = pattern_map.at(conv_pattern_m).get_node_shared_ptr();

        // The weight of the convolution node, [hidden_out, hidden_in, 1, 1]
        auto hidden_out = conv_node->input_value(1).get_partial_shape()[0].get_length();
        auto hidden_in = conv_node->input_value(1).get_partial_shape()[1].get_length();

        // The input of the convolution node, its shape should match conv_input_1/2/3 pattern
        std::vector<int64_t> input_transpose_order, output_transpose_order;
        if (pattern_map.count(conv_input_1)) {
            input_transpose_order = {2, 3, 0, 1};  // [seq_len, hidden_in, 1, 1] -> [1, 1, seq_len, hidden_in]
            output_transpose_order = {2, 3, 0, 1};
        } else if (pattern_map.count(conv_input_2)) {
            input_transpose_order = {0, 2, 3, 1};  // [1, hidden_in, 1, seq_len] -> [1, 1, seq_len, hidden_in]
            output_transpose_order = {0, 3, 1, 2};
        } else if (pattern_map.count(conv_input_3)) {
            input_transpose_order = {0, 3, 2, 1};  // [1, hidden_in, seq_len, 1] -> [1, 1, seq_len, hidden_in]
            output_transpose_order = {0, 3, 2, 1};
        } else {
            return false;
        }

        auto reshape_const_last_two_1 = [](std::shared_ptr<ov::Node> node) {
            auto constant = ov::as_type_ptr<ov::op::v0::Constant>(node);
            OPENVINO_ASSERT(constant != nullptr);
            ov::Shape current_shape = constant->get_shape();
            OPENVINO_ASSERT((current_shape.size() == 4 || current_shape.size() == 5) && current_shape[-1] == 1 &&
                            current_shape[-2] == 1);
            auto new_shape = current_shape.size() == 5 ? ov::Shape{current_shape[0], current_shape[1], current_shape[2]}
                                                       : ov::Shape{current_shape[0], current_shape[1]};
            auto new_constant = std::make_shared<ov::op::v0::Constant>(*constant, new_shape);
            ov::copy_weightless_cache_attr(constant, new_constant);
            return new_constant;
        };

        // Reshape compressed w/zp/scale constants, remove the last two dimensions of size 1
        std::shared_ptr<Node> weights_convert_new = nullptr;
        if (pattern_map.count(weights_5d_m)) {
            auto weights_5d = reshape_const_last_two_1(pattern_map.at(weights_5d_m).get_node_shared_ptr());
            auto weights_5d_convert = pattern_map.at(weights_5d_convert_m).get_node_shared_ptr();
            weights_convert_new = weights_5d_convert->clone_with_new_inputs({weights_5d});
            ov::copy_runtime_info(weights_5d_convert, weights_convert_new);
        } else {
            auto weights_4d = pattern_map.at(weights_4d_m).get_node_shared_ptr();
            weights_4d = reshape_const_last_two_1(weights_4d);
            auto weights_4d_reshape = pattern_map.at(weights_4d_reshape_m).get_node_shared_ptr();
            auto shape = weights_4d_reshape->get_output_shape(0);
            OPENVINO_ASSERT(shape.size() == 5 && shape[-1] == 1 && shape[-2] == 1);
            auto shape_new_constant =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                       ov::Shape{3},
                                                       std::vector<int64_t>{static_cast<int64_t>(shape[0]),
                                                                            static_cast<int64_t>(shape[1]),
                                                                            static_cast<int64_t>(shape[2])});
            auto weights_reshape_new = weights_4d_reshape->clone_with_new_inputs({weights_4d, shape_new_constant});
            ov::copy_runtime_info(weights_4d_reshape, weights_reshape_new);
            auto weights_4d_convert = pattern_map.at(weights_4d_convert_m).get_node_shared_ptr();
            weights_convert_new = weights_4d_convert->clone_with_new_inputs({weights_reshape_new});
            ov::copy_runtime_info(weights_4d_convert, weights_convert_new);
        }

        std::shared_ptr<Node> weights_sub_new = nullptr;
        if (pattern_map.count(weights_sub_m)) {
            std::shared_ptr<Node> zp_convert_new = nullptr;
            if (pattern_map.count(zp_5d_m)) {
                auto zp_5d = reshape_const_last_two_1(pattern_map.at(zp_5d_m).get_node_shared_ptr());
                auto zp_5d_convert = pattern_map.at(zp_5d_convert_m).get_node_shared_ptr();
                zp_convert_new = zp_5d_convert->clone_with_new_inputs({zp_5d});
                ov::copy_runtime_info(zp_5d_convert, zp_convert_new);
            } else {
                auto zp_4d = pattern_map.at(zp_4d_m).get_node_shared_ptr();
                zp_4d = reshape_const_last_two_1(zp_4d);
                auto zp_4d_unsqueeze = pattern_map.at(zp_4d_unsqueeze_m).get_node_shared_ptr();
                auto zp_unsqueeze_new =
                    zp_4d_unsqueeze->clone_with_new_inputs({zp_4d, zp_4d_unsqueeze->input_value(1)});
                ov::copy_runtime_info(zp_4d_unsqueeze, zp_unsqueeze_new);
                auto zp_4d_convert = pattern_map.at(zp_4d_convert_m).get_node_shared_ptr();
                zp_convert_new = zp_4d_convert->clone_with_new_inputs({zp_unsqueeze_new});
                ov::copy_runtime_info(zp_4d_convert, zp_convert_new);
            }
            auto weights_sub = pattern_map.at(weights_sub_m).get_node_shared_ptr();
            weights_sub_new = weights_sub->clone_with_new_inputs({weights_convert_new, zp_convert_new});
            ov::copy_runtime_info(weights_sub, weights_sub_new);
        } else {
            weights_sub_new = weights_convert_new;
        }

        std::shared_ptr<Node> scale_new = nullptr;
        if (pattern_map.count(scale_5d_m)) {
            scale_new = reshape_const_last_two_1(pattern_map.at(scale_5d_m).get_node_shared_ptr());
        } else {
            auto scale_4d = reshape_const_last_two_1(pattern_map.at(scale_4d_m).get_node_shared_ptr());
            auto scale_4d_unsqueeze = pattern_map.at(scale_4d_unsqueeze_m).get_node_shared_ptr();
            scale_new = scale_4d_unsqueeze->clone_with_new_inputs({scale_4d, scale_4d_unsqueeze->input_value(1)});
            ov::copy_runtime_info(scale_4d_unsqueeze, scale_new);
        }
        auto weights_mult = pattern_map.at(weights_mult_m).get_node_shared_ptr();
        auto weights_mult_new = weights_mult->clone_with_new_inputs({weights_sub_new, scale_new});
        ov::copy_runtime_info(weights_mult, weights_mult_new);

        // Reshape decompressed weights from 1x1 kernel [hidden_out, hidden_in, 1, 1] to matmul var b
        // [hidden_out, hidden_in]
        auto weights_new_constant = std::make_shared<ov::op::v0::Constant>(ov::element::i64,
                                                                           ov::Shape{2},
                                                                           std::vector<int64_t>{hidden_out, hidden_in});
        auto weights_reshape = pattern_map.at(weights_reshape_m).get_node_shared_ptr();
        auto weights_reshape_new = weights_reshape->clone_with_new_inputs({weights_mult_new, weights_new_constant});
        ov::copy_runtime_info(weights_reshape, weights_new_constant);
        ov::copy_runtime_info(weights_reshape, weights_reshape_new);

        // Transpose convolution input0 to [1, 1, seq_len, hidden_in]
        auto input = conv_node->input_value(0);
        auto input_transpose_const =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, input_transpose_order);
        auto transpose_input = std::make_shared<ov::op::v1::Transpose>(input, input_transpose_const);

        // MatMul: [1, 1, seq_len, hidden_in] x [hidden_out, hidden_in]^T => [1, 1, seq_len, hidden_out]
        auto matmul = std::make_shared<ov::op::v0::MatMul>(transpose_input, weights_reshape_new, false, true);

        // Add bias
        std::shared_ptr<ov::Node> bias_new = nullptr;
        if (pattern_map.count(bias_const_m)) {
            auto bias_const = ov::as_type_ptr<ov::op::v0::Constant>(pattern_map.at(bias_const_m).get_node_shared_ptr());
            bias_new = std::make_shared<ov::op::v0::Constant>(*bias_const,
                                                              ov::Shape{bias_const->get_shape()[0],
                                                                        bias_const->get_shape()[2],
                                                                        bias_const->get_shape()[3],
                                                                        bias_const->get_shape()[1]});
            ov::copy_runtime_info(bias_const, bias_new);
        } else {
            auto bias_input = pattern_map.at(bias_input_m).get_node_shared_ptr();
            auto bias_input_shape = bias_input->get_output_partial_shape(0);
            std::vector<int64_t> new_bias_shape = {bias_input_shape[0].get_length(),
                                                   bias_input_shape[2].get_length(),
                                                   bias_input_shape[3].get_length(),
                                                   bias_input_shape[1].get_length()};
            auto bias_shape_const =
                std::make_shared<ov::op::v0::Constant>(ov::element::i64, ov::Shape{4}, new_bias_shape);
            bias_new = std::make_shared<ov::op::v1::Reshape>(bias_input, bias_shape_const, false);
            ov::copy_runtime_info(bias_input, {bias_shape_const, bias_new});
        }

        auto bias_out = pattern_map.at(bias_out_m).get_node_shared_ptr();
        auto matmul_out = bias_out->clone_with_new_inputs({matmul, bias_new});
        ov::copy_runtime_info(bias_out, matmul_out);

        // Transpose convolutionoutput back to the original layout
        auto output_transpose_const =
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, output_transpose_order);
        auto final_node = std::make_shared<ov::op::v1::Transpose>(matmul_out, output_transpose_const);
        ov::copy_runtime_info(conv_node, {transpose_input, matmul, final_node});

        final_node->set_friendly_name(m.get_match_root()->get_friendly_name());
        ov::replace_node(m.get_match_root(), final_node);

        return true;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(bias_out_m, matcher_name);
    this->register_matcher(m, callback);
}
