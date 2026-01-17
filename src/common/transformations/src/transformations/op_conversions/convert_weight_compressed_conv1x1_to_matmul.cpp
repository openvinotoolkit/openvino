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

    auto weights_const_m = wrap_type<ov::op::v0::Constant>(rank_equals(4) && has_static_rank() && filter1x1_path);
    auto weights_param_m = wrap_type<ov::op::v0::Parameter>(rank_equals(4) && has_static_rank() && filter1x1_path);
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
    auto conv1x1_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution>({a_m, weight_mult_m});

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
            } else {
                OPENVINO_ASSERT(current_shape.size() == 4);
                OPENVINO_ASSERT(current_shape[2] == 1 && current_shape[3] == 1);

                auto new_shape = ov::Shape{current_shape[0], current_shape[1]};

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

            auto reshape_weight_const = ov::op::v0::Constant::create(element::i32, Shape{2}, values_reshape_b);
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
