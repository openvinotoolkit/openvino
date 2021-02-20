// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/or.hpp>
#include <ngraph/rt_info.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/common_optimizations/weights_dequantize_to_fake_quantize.hpp>
#include "itt.hpp"

NGRAPH_RTTI_DEFINITION(ngraph::pass::WeightsDequantizeToFakeQuantize, "WeightsDequantizeToFakeQuantize", 0);

ngraph::pass::WeightsDequantizeToFakeQuantize::WeightsDequantizeToFakeQuantize() {
    MATCHER_SCOPE(WeightsDequantizeToFakeQuantize);

    const auto weights = ngraph::pattern::wrap_type<ngraph::opset6::Constant>(pattern::type_matches(element::i8));
    const auto convert = ngraph::pattern::wrap_type<ngraph::opset6::Convert>({weights});
    const auto sub_c = ngraph::pattern::wrap_type<ngraph::opset6::Constant>();
    const auto sub = ngraph::pattern::wrap_type<ngraph::opset6::Subtract>({convert, sub_c});

    const auto sub_or_convert = std::make_shared<pattern::op::Or>(OutputVector{convert, sub});

    const auto mul_c = ngraph::pattern::wrap_type<ngraph::opset6::Constant>();
    const auto mul = ngraph::pattern::wrap_type<ngraph::opset6::Multiply>({sub_or_convert, mul_c});

    ngraph::matcher_pass_callback callback;
    callback = [=](ngraph::pattern::Matcher &m) {
        const auto &pattern_map = m.get_pattern_map();

        const auto &weights_node = as_type_ptr<opset6::Constant>(pattern_map.at(weights));
        const auto &convert_node = pattern_map.at(convert);
        const auto &multiply_node = pattern_map.at(mul);
        const auto &scale_node = pattern_map.at(mul_c);
        if (!weights_node || !convert_node || !multiply_node || !scale_node) {
            return false;
        }

        const auto *data = weights_node->get_data_ptr<int8_t>();
        const int8_t weights_minimum = *std::min_element(data, data + shape_size(weights_node->get_shape()));
        int64_t levels = (weights_minimum == static_cast<int8_t>(-128)) ? 256 : 255;
        int64_t in_low = -(levels / 2), in_high = levels + in_low - 1;

        const auto &input_low = opset6::Constant::create(convert_node->get_element_type(), {}, {in_low});
        const auto &input_high = opset6::Constant::create(convert_node->get_element_type(), {}, {in_high});

        auto &zero_point = pattern_map.count(sub_c) ? pattern_map.at(sub_c) : opset6::Constant::create(convert_node->get_element_type(), {}, {0});

        const auto &output_low = op::util::eltwise_fold<opset6::Multiply>(
                op::util::eltwise_fold<opset6::Subtract>(input_low, zero_point), scale_node);
        const auto &output_high = op::util::eltwise_fold<opset6::Multiply>(
                op::util::eltwise_fold<opset6::Subtract>(input_high, zero_point), scale_node);

        auto fq = std::make_shared<opset6::FakeQuantize>(
                convert_node, input_low, input_high, output_low, output_high, levels);

        NodeVector nodes_to_copy_RT_info_from{multiply_node, scale_node, zero_point};
        if (pattern_map.count(sub))
            nodes_to_copy_RT_info_from.push_back(sub);

        ngraph::copy_runtime_info(nodes_to_copy_RT_info_from, fq);
        multiply_node->output(0).replace(fq->output(0));

        if (convert_node->get_rt_info().count("DISABLED_CONSTANT_FOLDING"))
            convert_node->get_rt_info().erase("DISABLED_CONSTANT_FOLDING");
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(mul, matcher_name);
    register_matcher(m, callback);
}
