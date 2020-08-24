// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_quantize_dequantize.hpp"
#include "transformations/utils/utils.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


// ConvertQuantizeDequantize converts Quantize/Dequantize pair to a single FakeQuantize.
// Since Quantize is decomposed to FakeQuantize and Dequantize is decomposed to Subtract->Multiply,
// the full pattern to match is presented on the left hand side of the graph below.
// On the right hand side is the graph after transformation.
// Currently transformation supports only i8 and u8 quantized data type.
// That implies 'levels' attribute to be 256, as well as (output_low, output_high) be (-128, 127) or (0, 255) (depends on sign of the quantized data type).
// Another limitation is that 'zero_point' and 'scale' have to be broadcastable to the output of FakeQuantize.
//
//
//                                   |  |  |  |  |
//                                   |  |  |  |  |
//                                   v  v  v  v  v
//                                  +------------+
//                                  |FakeQuantize|
//                                  +------------+
//                                        |
//                                        v
//                              +---------------------+
//                              |      Convert        |
//                              |(e.g. from f32 to u8)|
//                              +---------+-----------+                            |  |  |  |  |
//                                        |                                        |  |  |  |  |
//                                        v                                        v  v  v  v  v
//                              +---------------------+                           +------------+
//                              |      Convert        |            ====>          |FakeQuantize|
//                              |  (from u8 to f32)   |                           +------------+
//                              +---------+-----------+                                 |
//                                        |                                             v
//                                        v
//                  +----------+    +------------+
//                  |zero point|--->|  Subtract  |
//                  +----------+    +-----+------+
//                                        |
//                                        v
//                   +---------+    +------------+
//                   |  scale  |--->|  Multiply  |
//                   +---------+    +-----+------+
//                                        |
//                                        v
//


ngraph::pass::ConvertQuantizeDequantize::ConvertQuantizeDequantize() {
    auto data_pattern = ngraph::pattern::any_input();
    auto input_low_pattern = ngraph::pattern::any_input();
    auto input_high_pattern = ngraph::pattern::any_input();
    auto output_low_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto output_high_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto fq_pattern = ngraph::pattern::wrap_type<opset1::FakeQuantize>({data_pattern, input_low_pattern,
                                                                       input_high_pattern, output_low_pattern,
                                                                       output_high_pattern});
    auto convert1_pattern = ngraph::pattern::wrap_type<opset1::Convert>({fq_pattern}, pattern::type_matches_any({element::i8, element::u8}));
    auto convert2_pattern = ngraph::pattern::wrap_type<opset1::Convert>({convert1_pattern}, pattern::type_matches(element::f32));
    auto zero_point_pattern = ngraph::pattern::any_input();
    auto sub_pattern = ngraph::pattern::wrap_type<opset1::Subtract>({convert2_pattern, zero_point_pattern}, pattern::consumers_count(1));
    auto scale_pattern = ngraph::pattern::any_input();
    auto mul_pattern = ngraph::pattern::wrap_type<opset1::Multiply>({sub_pattern, scale_pattern});

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        auto pattern_map = m.get_pattern_map();
        auto data = pattern_map[data_pattern];
        auto input_low = pattern_map[input_low_pattern];
        auto input_high = pattern_map[input_high_pattern];
        auto output_low = std::dynamic_pointer_cast<opset1::Constant>(pattern_map[output_low_pattern]);
        auto output_high = std::dynamic_pointer_cast<opset1::Constant>(pattern_map[output_high_pattern]);
        auto fq = std::dynamic_pointer_cast<opset1::FakeQuantize>(pattern_map[fq_pattern]);
        auto zero_point = pattern_map[zero_point_pattern];
        auto scale = pattern_map[scale_pattern];
        auto convert1 = pattern_map[convert1_pattern];
        auto convert2 = pattern_map[convert2_pattern];
        auto sub = pattern_map[sub_pattern];
        auto mul = pattern_map[mul_pattern];

        size_t levels = fq->get_levels();
        if (levels != 256)
            return false;

        float out_low_val;
        if (!op::util::get_single_value(output_low, out_low_val))
            return false;
        float out_high_val;
        if (!op::util::get_single_value(output_high, out_high_val))
            return false;
        const auto& type = convert1->get_element_type();
        switch (type) {
        case element::Type_t::i8:
            if (out_low_val != -128 || out_high_val != 127)
                return false;
            break;
        case element::Type_t::u8:
            if (out_low_val != 0 || out_high_val != 255)
                return false;
            break;
        default:
            return false;
        }

        auto new_out_low = std::make_shared<ngraph::opset1::Multiply>(
                std::make_shared<ngraph::opset1::Subtract>(output_low, zero_point), scale);
        auto new_out_high = std::make_shared<ngraph::opset1::Multiply>(
                std::make_shared<ngraph::opset1::Subtract>(output_high, zero_point), scale);

        // check if new_out_low/high shapes are broadcastable to FQ's input
        auto data_shape = data->get_output_partial_shape(0);
        auto out_low_shape = new_out_low->get_output_partial_shape(0);
        if (out_low_shape.rank().get_length() > data_shape.rank().get_length())
            return false;
        auto out_high_shape = new_out_high->get_output_partial_shape(0);
        if (out_high_shape.rank().get_length() > data_shape.rank().get_length())
            return false;

        auto fake_q = std::make_shared<ngraph::opset1::FakeQuantize>(data, input_low, input_high, new_out_low, new_out_high, levels);
        fake_q->set_friendly_name(mul->get_friendly_name());

        copy_runtime_info({data, input_low, input_high, output_low, output_high, fq, zero_point, scale, convert1, convert2, sub, mul},
                          {data, input_low, input_high, new_out_low, new_out_high, fake_q});
        replace_node(mul, fake_q);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_pattern, "ConvertQuantizeDequantize");
    this->register_matcher(m, callback);
}
