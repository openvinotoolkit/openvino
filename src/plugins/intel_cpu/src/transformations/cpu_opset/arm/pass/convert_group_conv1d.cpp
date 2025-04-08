// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0

#include "convert_group_conv1d.hpp"

#include <numeric>
#include <openvino/opsets/opset8.hpp>

#include "openvino/core/rt_info.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/pattern/op/wrap_type.hpp"

template <class Conv>
ov::matcher_pass_callback ov::intel_cpu::ConvertConv1DBase::convert_conv1d_to_conv2d() {
    return [&](ov::pass::pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<Conv>(m.get_match_root());
        if (!conv) {
            return false;
        }

        const auto& input0 = conv->input_value(0);
        const auto& input_shape = input0.get_partial_shape();
        // is Conv1D
        if (input_shape.size() != 3) {
            return false;
        }

        auto input = conv->input_value(0);
        auto weights = conv->input_value(1);

        auto weights2d_shape = weights.get_shape();
        weights2d_shape.push_back(1);
        auto w_shape = std::make_shared<ov::opset8::Constant>(ov::element::i64,
                                                              ov::Shape{weights2d_shape.size()},
                                                              weights2d_shape);

        auto getUnsqueeze = [&](const ov::Output<ov::Node>& node) {
            auto rank = node.get_partial_shape().rank().get_length();
            return std::make_shared<ov::opset8::Unsqueeze>(
                node,
                ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {rank}));
        };

        auto input2d = getUnsqueeze(input);
        auto weights2d = getUnsqueeze(weights);

        auto conv2d = std::make_shared<Conv>(input2d,
                                             weights2d,
                                             ov::Strides{conv->get_strides()[0], 1},
                                             ov::CoordinateDiff{conv->get_pads_begin()[0], 0},
                                             ov::CoordinateDiff{conv->get_pads_end()[0], 0},
                                             ov::Strides{conv->get_dilations()[0], 1},
                                             conv->get_auto_pad());

        auto reshape = std::make_shared<ov::opset8::Squeeze>(
            conv2d,
            ov::opset1::Constant::create(ov::element::i64, ov::Shape{1}, {input_shape.rank().get_length()}));

        reshape->set_friendly_name(conv->get_friendly_name());
        ov::copy_runtime_info(conv, {input2d, weights2d, conv2d, reshape});
        ov::replace_node(conv, reshape);
        return true;
    };
}

ov::intel_cpu::ConvertConv1D::ConvertConv1D() {
    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        ov::pass::pattern::wrap_type<ov::opset8::Convolution>(
            {ov::pass::pattern::any_input(), ov::pass::pattern::any_input()}),
        "ConvertConvolutionToArm");
    register_matcher(m, convert_conv1d_to_conv2d<ov::opset8::Convolution>());
}

ov::intel_cpu::ConvertGroupConv1D::ConvertGroupConv1D() {
    auto m = std::make_shared<ov::pass::pattern::Matcher>(
        ov::pass::pattern::wrap_type<ov::opset8::GroupConvolution>(
            {ov::pass::pattern::any_input(), ov::pass::pattern::any_input()}),
        "ConvertGroupConvolutionToArm");
    register_matcher(m, convert_conv1d_to_conv2d<ov::opset8::GroupConvolution>());
}