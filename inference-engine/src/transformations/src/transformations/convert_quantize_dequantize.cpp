// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_quantize_dequantize.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>


ngraph::pass::ConvertQuantizeDequantize::ConvertQuantizeDequantize() {
    auto data_pattern = ngraph::pattern::any_input();
    auto input_low_pattern = ngraph::pattern::any_input();
    auto input_high_pattern = ngraph::pattern::any_input();
    auto output_low_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto output_high_pattern = ngraph::pattern::wrap_type<opset1::Constant>();
    auto fq_pattern = ngraph::pattern::wrap_type<opset1::FakeQuantize>({data_pattern, input_low_pattern,
                                                                       input_high_pattern, output_low_pattern,
                                                                       output_high_pattern});
    auto convert1_pattern = ngraph::pattern::wrap_type<opset1::Convert>({fq_pattern});
    auto convert2_pattern = ngraph::pattern::wrap_type<opset1::Convert>({convert1_pattern});
    auto zero_point_pattern = ngraph::pattern::any_input();
    auto sub_pattern = ngraph::pattern::wrap_type<opset1::Subtract>({convert2_pattern, zero_point_pattern}, pattern::consumers_count(1));
    auto scale_pattern = ngraph::pattern::any_input();
    auto mul_pattern = ngraph::pattern::wrap_type<opset1::Multiply>({sub_pattern, scale_pattern});

    ngraph::graph_rewrite_callback callback = [=](pattern::Matcher& m) {
        auto mul = std::dynamic_pointer_cast<ngraph::opset1::Multiply> (m.get_match_root());
        if (!mul)
            return false;

        auto pattern_map = m.get_pattern_map();

        auto convert2 = pattern_map[convert2_pattern];
        NGRAPH_CHECK(convert2->get_element_type() == element::f32);

        auto fq = std::dynamic_pointer_cast<opset1::FakeQuantize>(pattern_map[fq_pattern]);
        NGRAPH_CHECK(fq != nullptr);
        size_t levels = fq->get_levels();
        NGRAPH_CHECK(levels == 256);

        auto output_low = pattern_map[output_low_pattern];
        auto output_high = pattern_map[output_high_pattern];
        auto output_low_const = std::dynamic_pointer_cast<opset1::Constant>(output_low);
        auto output_high_const = std::dynamic_pointer_cast<opset1::Constant>(output_high);
        auto out_low_val = output_low_const->cast_vector<float>();
        auto out_high_val = output_high_const->cast_vector<float>();
        NGRAPH_CHECK(out_low_val.size() == 1 && out_high_val.size() == 1);

        auto convert1 = pattern_map[convert1_pattern];
        const auto& type = convert1->get_element_type();
        switch (type) {
        case element::Type_t::i8:
            if (out_low_val[0] != -128 || out_high_val[0] != 127)
                return false;
            break;
        case element::Type_t::u8:
            if (out_low_val[0] != 0 || out_high_val[0] != 255)
                return false;
            break;
        default:
            NGRAPH_CHECK(false, "not supported data type" + type.get_type_name());
        }

        auto zero_point = pattern_map[zero_point_pattern];
        auto scale = pattern_map[scale_pattern];
        auto new_out_low = std::make_shared<ngraph::opset1::Multiply>(
                std::make_shared<ngraph::opset1::Subtract>(output_low, zero_point), scale);
        auto new_out_high = std::make_shared<ngraph::opset1::Multiply>(
                std::make_shared<ngraph::opset1::Subtract>(output_high, zero_point), scale);
        auto data = pattern_map[data_pattern];
        auto input_low = pattern_map[input_low_pattern];
        auto input_high = pattern_map[input_high_pattern];
        auto fake_q = std::make_shared<ngraph::opset1::FakeQuantize>(data, input_low, input_high, new_out_low, new_out_high, levels);
        fake_q->set_friendly_name(mul->get_friendly_name());

        copy_runtime_info(mul, {data, input_low, input_high, new_out_low, new_out_high, fake_q});
        replace_node(mul, fake_q);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul_pattern, "ConvertQuantizeDequantize");
    this->register_matcher(m, callback);
}
