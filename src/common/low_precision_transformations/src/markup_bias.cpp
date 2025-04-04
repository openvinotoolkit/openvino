// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_bias.hpp"

#include <memory>
#include "openvino/pass/pattern/op/wrap_type.hpp"

#include "itt.hpp"
#include "low_precision/rt_info/bias_attribute.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/convolution.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/group_conv.hpp"
#include "openvino/op/matmul.hpp"

using namespace ov::pass::low_precision;

MarkupBias::MarkupBias() {
    MATCHER_SCOPE(MarkupBias);
    auto layer_m = ov::pass::pattern::wrap_type<ov::op::v1::Convolution,
                                                ov::op::v1::GroupConvolution,
                                                ov::op::v1::ConvolutionBackpropData,
                                                ov::op::v1::GroupConvolutionBackpropData,
                                                ov::op::v0::MatMul>(ov::pass::pattern::has_static_rank());
    auto bias_const_m = ov::pass::pattern::wrap_type<ov::op::v0::Constant>();
    auto bias_m = ov::pass::pattern::wrap_type<ov::op::v1::Add>({layer_m, bias_const_m});

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
