// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/validation_util.hpp>
#include <ngraph/rt_info.hpp>

#include "transformations/common_optimizations/compress_quantize_weights.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::CompressQuantizeWeights, "CompressQuantizeWeights", 0);

ngraph::pass::CompressQuantizeWeights::CompressQuantizeWeights() {
    auto weights_pattern = pattern::wrap_type<opset8::Constant>();
    auto input_low_pattern = pattern::wrap_type<opset8::Constant>();
    auto input_high_pattern = pattern::wrap_type<opset8::Constant>();
    auto output_low_pattern = pattern::wrap_type<opset8::Constant>();
    auto output_high_pattern = pattern::wrap_type<opset8::Constant>();
    auto fq_pattern = pattern::wrap_type<opset8::FakeQuantize>({weights_pattern, input_low_pattern, input_high_pattern,
                                                                output_low_pattern, output_high_pattern});

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        auto fq = std::dynamic_pointer_cast<opset8::FakeQuantize>(m.get_match_root());
        if (!fq)
            return false;
        auto levels = fq->get_levels();
        auto quantized_type = element::undefined;
        // Currently we support two weights quantize types: i4 and i8
        switch (levels) {
            case 16:
                quantized_type = element::i4;
                break;
            case 256:
                quantized_type = element::i8;
                break;
            default:
                return false;
        }

        const auto& pattern_value_map = m.get_pattern_value_map();
        const auto& input_type = fq->get_element_type();

        /*
           Quantize part

           Prepare new FakeQuantize that performs weights quantization.
           In this case input_low/high stays the same, but we need new output_low/high:
             output_low = -levels / 2
             output_high = levels / 2 - 1
           The FakeQuantize result is converted to low precision type and then constant folded
        */
        std::shared_ptr<Node> new_input_low;
        auto new_output_low = op::Constant::create(input_type, Shape{}, {-static_cast<float>(levels) / 2});
        auto new_output_high = op::Constant::create(input_type, Shape{}, {static_cast<float>(levels) / 2 - 1});
        const auto& weights = pattern_value_map.at(weights_pattern);
        const auto& input_low = pattern_value_map.at(input_low_pattern);
        const auto& input_high = pattern_value_map.at(input_high_pattern);
        auto quantize = fq->clone_with_new_inputs({weights, input_low, input_high,
                                                   new_output_low, new_output_high});
        // Convert quantized weights to low precision type
        std::shared_ptr<Node> convert_to_low_prec = std::make_shared<opset8::Convert>(quantize, quantized_type);
        // Constant fold quantized weights
        if (auto constant = get_constant_from_source(convert_to_low_prec)) {
            convert_to_low_prec = constant;
        } else {
            return false;
        }
        convert_to_low_prec->set_friendly_name(weights.get_node()->get_friendly_name());

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
                scale = (output_high - output_low) / (input_high - input_low)
                zero_point = input_low - output_low / scale
        */
        const auto& output_low = pattern_value_map.at(output_low_pattern);
        const auto& output_high = pattern_value_map.at(output_high_pattern);
        auto output_range = std::make_shared<opset8::Subtract>(output_high, output_low);
        auto input_range = std::make_shared<opset8::Subtract>(input_high, input_low);
        std::shared_ptr<Node> scale = std::make_shared<opset8::Divide>(output_range, input_range);
        auto descaled_output_low = std::make_shared<opset8::Divide>(output_low, scale);
        std::shared_ptr<Node> shift = std::make_shared<opset8::Subtract>(input_low, descaled_output_low);
        if (auto constant = get_constant_from_source(scale))
            scale = constant;
        auto zero = op::Constant::create(input_type, Shape{}, {0});
        auto scale_eq_zero = std::make_shared<opset8::Equal>(scale, zero);
        // shift equals to input_low - output_low / scale
        // for positions where scale == 0, we put zero as shift
        std::shared_ptr<Node> zero_point = std::make_shared<opset8::Select>(scale_eq_zero, zero, shift);
        if (auto constant = get_constant_from_source(zero_point))
            zero_point = constant;
        if (auto constant = get_constant_from_source(scale))
            scale = constant;
        auto convert_to_high_prec = std::make_shared<opset8::Convert>(convert_to_low_prec, input_type);
        auto sub = std::make_shared<opset8::Subtract>(convert_to_high_prec, zero_point);
        auto mul = register_new_node<opset8::Multiply>(sub, scale);
        mul->set_friendly_name(fq->get_friendly_name());
        copy_runtime_info(fq, {convert_to_high_prec, sub, mul});
        replace_node(fq, mul);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(fq_pattern, "CompressQuantizeWeights");
    this->register_matcher(m, callback);
}
