// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_bias.hpp"

#include <memory>
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "itt.hpp"
#include "low_precision/rt_info/bias_attribute.hpp"

using namespace ov::pass::low_precision;

MarkupBias::MarkupBias() {
    MATCHER_SCOPE(MarkupBias);
    auto layer_m = ov::pass::pattern::wrap_type<ov::opset1::Convolution,
                                                ov::opset1::GroupConvolution,
                                                ov::opset1::ConvolutionBackpropData,
                                                ov::opset1::GroupConvolutionBackpropData,
                                                ov::opset1::MatMul>(ov::pass::pattern::has_static_rank());
    auto bias_const_m = ov::pass::pattern::wrap_type<ov::opset1::Constant>();
    auto bias_m = ov::pass::pattern::wrap_type<ov::opset1::Add>({layer_m, bias_const_m});

    ov::matcher_pass_callback callback = [=](ov::pass::pattern::Matcher& m) {
        const auto& pattern_map = m.get_pattern_value_map();
        const auto& const_shape = pattern_map.at(bias_const_m).get_shape();

        const bool per_channel = std::count_if(const_shape.begin(), const_shape.end(), [](size_t x) { return x > 1; }) == 1;
        if (ov::shape_size(const_shape) == 1 || per_channel) {
            const auto bias = pattern_map.at(bias_m).get_node_shared_ptr();
            ov::mark_as_bias(bias);
        }

        return false;
    };

    auto m = std::make_shared<ov::pass::pattern::Matcher>(bias_m, matcher_name);
    register_matcher(m, callback);
}
