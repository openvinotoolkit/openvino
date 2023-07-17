// Copyright (C) 2020-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0


#include "convert_group_conv1d.hpp"

#include <numeric>

#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset8.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

template <class Conv>
ngraph::matcher_pass_callback ov::intel_cpu::ConvertConv1DBase::convert_conv1d_to_conv2d() {
    return [&](ngraph::pattern::Matcher& m) {
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

        auto input   = conv->input_value(0);
        auto weights = conv->input_value(1);

        auto weights2d_shape = weights.get_shape();
        weights2d_shape.push_back(1);
        auto w_shape = std::make_shared<ov::opset8::Constant>(ngraph::element::i64, ngraph::Shape{weights2d_shape.size()}, weights2d_shape);

        auto getUnsqueeze = [&](const ngraph::Output<ngraph::Node>& node) {
            auto rank = node.get_partial_shape().rank().get_length();
            return std::make_shared<ov::opset8::Unsqueeze>(node,
                                                           ov::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {rank}));
        };

        auto input2d = getUnsqueeze(input);
        auto weights2d = getUnsqueeze(weights);

        auto conv2d = std::make_shared<Conv>(input2d,
                                             weights2d,
                                             ngraph::Strides{conv->get_strides()[0], 1},
                                             ngraph::CoordinateDiff{conv->get_pads_begin()[0], 0},
                                             ngraph::CoordinateDiff{conv->get_pads_end()[0], 0},
                                             ngraph::Strides{conv->get_dilations()[0], 1},
                                             conv->get_auto_pad());

        auto reshape = std::make_shared<ov::opset8::Squeeze>(
            conv2d,
            ov::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {input_shape.rank().get_length()}));

        reshape->set_friendly_name(conv->get_friendly_name());
        ngraph::copy_runtime_info(conv, {input2d, weights2d, conv2d, reshape});
        ngraph::replace_node(conv, reshape);
        return true;
    };
}

ov::intel_cpu::ConvertConv1D::ConvertConv1D() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
        ngraph::pattern::wrap_type<ov::opset8::Convolution>({ngraph::pattern::any_input(),
                                                             ngraph::pattern::any_input()}),
                                                            "ConvertConvolutionToArm");
    register_matcher(m, convert_conv1d_to_conv2d<ov::opset8::Convolution>());
}

ov::intel_cpu::ConvertGroupConv1D::ConvertGroupConv1D() {
    auto m = std::make_shared<ngraph::pattern::Matcher>(
            ngraph::pattern::wrap_type<ov::opset8::GroupConvolution>({ngraph::pattern::any_input(),
                                                                      ngraph::pattern::any_input()}),
                                                                     "ConvertGroupConvolutionToArm");
    register_matcher(m, convert_conv1d_to_conv2d<ov::opset8::GroupConvolution>());
}