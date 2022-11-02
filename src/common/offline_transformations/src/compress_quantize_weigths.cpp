// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <compress_quantize_weights.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/validation_util.hpp>
#include <openvino/pass/constant_folding.hpp>

static bool has_dequantization_subgraph(const std::shared_ptr<ngraph::Node>& first_convert) {
    auto first_convert_users = first_convert->get_users();
    const auto second_convert = std::find_if(first_convert_users.begin(),
                                             first_convert_users.end(),
                                             [](const std::shared_ptr<ngraph::Node>& n) -> bool {
                                                 return ov::is_type<ngraph::opset8::Convert>(n);
                                             });
    if (second_convert == first_convert_users.end())
        return false;
    auto convert_or_subtract_users = (*second_convert)->get_users();
    const auto subtract = std::find_if(convert_or_subtract_users.begin(),
                                       convert_or_subtract_users.end(),
                                       [](const std::shared_ptr<ngraph::Node>& n) -> bool {
                                           return ov::is_type<ngraph::opset8::Subtract>(n);
                                       });
    if (subtract != convert_or_subtract_users.end()) {
        convert_or_subtract_users = (*subtract)->get_users();
    }
    const auto multiply = std::find_if(convert_or_subtract_users.begin(),
                                       convert_or_subtract_users.end(),
                                       [](const std::shared_ptr<ngraph::Node>& n) -> bool {
                                           return ov::is_type<ngraph::opset8::Multiply>(n);
                                       });
    return multiply != convert_or_subtract_users.end();
}

ngraph::pass::CompressQuantizeWeights::CompressQuantizeWeights() {
    auto weights_pattern = pattern::wrap_type<opset8::Constant>();
    auto input_low_pattern = pattern::wrap_type<opset8::Constant>();
    auto input_high_pattern = pattern::wrap_type<opset8::Constant>();
    auto output_low_pattern = pattern::wrap_type<opset8::Constant>();
    auto output_high_pattern = pattern::wrap_type<opset8::Constant>();
    auto fq_pattern = pattern::wrap_type<opset8::FakeQuantize>(
        {weights_pattern, input_low_pattern, input_high_pattern, output_low_pattern, output_high_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto fq = std::dynamic_pointer_cast<opset8::FakeQuantize>(m.get_match_root());
        if (!fq)
            return false;
        auto levels = fq->get_levels();
        if (levels <= 2 || levels > 256)
            return false;
        auto quantized_type = element::undefined;
        // Currently we support two weights quantize types: i4 and i8
        if (levels <= 16) {
            quantized_type = element::i4;
        } else if (levels <= 256) {
            quantized_type = element::i8;
        }

        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& input_type = fq->get_element_type();

        // skip dequantize part if there is already dequantization subgraph after FakeQuantize
        auto fq_users = fq->get_users();
        if (fq_users.size() == 1 && has_dequantization_subgraph(fq_users[0])) {
            auto& first_convert = fq_users[0];
            if (auto new_weights = ov::get_constant_from_source(first_convert)) {
                replace_node(first_convert, new_weights);
                // preserve dequantization subgraph for LP transformations
                auto weights_users = new_weights->get_users();
                if (weights_users.size() == 1 && ov::is_type<ngraph::opset8::Convert>(weights_users[0])) {
                    ov::pass::disable_constant_folding(weights_users[0]);
                }
                return true;
            } else {
                return false;
            }
        } else {
            /*
               Quantize part

               Prepare new FakeQuantize that performs weights quantization.
               In this case input_low/high stays the same, but we need new output_low/high:
                 output_low = -levels / 2
                 output_high = levels - 1 + output_low
               The FakeQuantize result is converted to low precision type and then constant folded
            */
            std::shared_ptr<Node> new_input_low;
            auto new_output_low = op::Constant::create(input_type, Shape{}, {-static_cast<float>(levels / 2)});
            auto new_output_high =
                std::make_shared<opset8::Add>(new_output_low, op::Constant::create(input_type, Shape{}, {levels - 1}));
            const auto& weights = pattern_value_map.at(weights_pattern);
            const auto& input_low = pattern_value_map.at(input_low_pattern);
            const auto& input_high = pattern_value_map.at(input_high_pattern);
            auto quantize =
                fq->clone_with_new_inputs({weights, input_low, input_high, new_output_low, new_output_high});
            // Convert quantized weights to low precision type
            std::shared_ptr<Node> new_weights = std::make_shared<opset8::Convert>(quantize, quantized_type);
            // Constant fold quantized weights
            if (auto constant = ov::get_constant_from_source(new_weights)) {
                new_weights = constant;
            } else {
                return false;
            }
            new_weights->set_friendly_name(weights.get_node()->get_friendly_name());

            /*
               Dequantize part is performed by Convert(from low to high precision)->Subtract->Multiply subgraph.

                                 +-------------------------+
                                 |         Convert         |
                                 | (from low to high prec) |
                                 +-------------------------+
                                              |
                                              v
                        +----------+    +------------+
                        |zero point|--->|  Subtract  |
                        +----------+    +-----+------+
                                              |
                                              v
                         +---------+    +------------+
                         |  scale  |--->|  Multiply  |
                         +---------+    +-----+------+
                                              |
                                              v

                where:
                    scale = (output_high - output_low) / (new_output_high - new_output_low)
                    zero_point = new_output_low - output_low / scale
            */
            const auto& output_low = pattern_value_map.at(output_low_pattern);
            const auto& output_high = pattern_value_map.at(output_high_pattern);
            auto output_range = std::make_shared<opset8::Subtract>(output_high, output_low);
            auto input_range = std::make_shared<opset8::Subtract>(new_output_high, new_output_low);
            std::shared_ptr<Node> scale = std::make_shared<opset8::Divide>(output_range, input_range);
            auto descaled_output_low = std::make_shared<opset8::Divide>(output_low, scale);
            std::shared_ptr<Node> shift = std::make_shared<opset8::Subtract>(new_output_low, descaled_output_low);
            if (auto constant = ov::get_constant_from_source(scale))
                scale = constant;
            auto zero = op::Constant::create(input_type, Shape{}, {0});
            auto scale_eq_zero = std::make_shared<opset8::Equal>(scale, zero);
            // shift equals to input_low - output_low / scale
            // for positions where scale == 0, we put zero as shift
            std::shared_ptr<Node> zero_point = std::make_shared<opset8::Select>(scale_eq_zero, zero, shift);
            if (auto constant = ov::get_constant_from_source(zero_point))
                zero_point = constant;
            if (auto constant = ov::get_constant_from_source(scale))
                scale = constant;
            auto convert_to_high_prec = std::make_shared<opset8::Convert>(new_weights, input_type);
            auto sub = register_new_node<opset8::Subtract>(convert_to_high_prec, zero_point);
            auto mul = register_new_node<opset8::Multiply>(sub, scale);
            mul->set_friendly_name(fq->get_friendly_name());
            copy_runtime_info(fq, {convert_to_high_prec, sub, mul});
            ov::pass::disable_constant_folding(convert_to_high_prec);
            replace_node(fq, mul);
        }
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_pattern, "CompressQuantizeWeights");
    this->register_matcher(m, callback);
}

ngraph::pass::ZeroPointOptimizer::ZeroPointOptimizer() {
    auto weights_pattern = pattern::wrap_type<opset8::Constant>();
    auto zero_point_pattern = pattern::wrap_type<opset8::Constant>();
    auto convert_pattern = pattern::wrap_type<opset8::Convert>({weights_pattern});
    auto sub_pattern = pattern::wrap_type<opset8::Subtract>({convert_pattern, zero_point_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        const auto& pattern_value_map = m.get_pattern_value_map();
        auto convert = pattern_value_map.at(convert_pattern).get_node_shared_ptr();
        auto sub = pattern_value_map.at(sub_pattern).get_node_shared_ptr();
        auto weights =
            std::dynamic_pointer_cast<opset8::Constant>(pattern_value_map.at(weights_pattern).get_node_shared_ptr());
        if (!weights || weights->get_element_type() != element::i8)
            return false;
        auto zero_point =
            std::dynamic_pointer_cast<opset8::Constant>(pattern_value_map.at(zero_point_pattern).get_node_shared_ptr());
        if (!zero_point)
            return false;

        auto zp_value = zero_point->cast_vector<float>();
        if (std::all_of(zp_value.begin(), zp_value.end(), [](float f) -> bool {
                return std::fabs(f) <= std::numeric_limits<float>::epsilon();
            })) {
            copy_runtime_info(sub, convert);
            replace_node(sub, convert);
        }

        auto int8_zero_point = std::make_shared<opset8::Convert>(
            std::make_shared<opset8::Round>(zero_point, opset8::Round::RoundMode::HALF_TO_EVEN),
            weights->get_element_type());
        auto adj_zero_point = std::make_shared<opset8::Subtract>(
            zero_point,
            std::make_shared<opset8::Convert>(int8_zero_point, convert->get_element_type()));

        auto adj_zero_point_const = ov::get_constant_from_source(adj_zero_point);
        if (!adj_zero_point_const)
            return false;
        auto adj_zero_point_val = adj_zero_point_const->cast_vector<float>();
        bool is_adj_zero_point_close_to_zero =
            std::all_of(adj_zero_point_val.begin(), adj_zero_point_val.end(), [](float f) -> bool {
                return std::fabs(f) < 1e-4;
            });
        if (!is_adj_zero_point_close_to_zero)
            return false;

        auto transformed = std::make_shared<opset8::Subtract>(
            std::make_shared<opset8::Convert>(std::make_shared<opset8::Subtract>(weights, int8_zero_point),
                                              convert->get_element_type()),
            adj_zero_point);
        auto diff = std::make_shared<opset8::Subtract>(sub, transformed);
        auto diff_const = ov::get_constant_from_source(diff);
        if (!diff_const)
            return false;
        auto diff_val = diff_const->cast_vector<float>();
        bool is_transformed_and_original_equal = std::all_of(diff_val.begin(), diff_val.end(), [](float f) -> bool {
            return std::fabs(f) < std::numeric_limits<float>::epsilon();
        });
        if (!is_transformed_and_original_equal)
            return false;

        std::shared_ptr<Node> new_weights = std::make_shared<opset8::Subtract>(weights, int8_zero_point);
        if (auto constant = ov::get_constant_from_source(new_weights))
            new_weights = constant;
        else
            return false;
        new_weights->set_friendly_name(weights->get_friendly_name());
        replace_node(weights, new_weights);

        copy_runtime_info(sub, convert);
        replace_node(sub, convert);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sub_pattern, "ZeroPointOptimizer");
    this->register_matcher(m, callback);
}
