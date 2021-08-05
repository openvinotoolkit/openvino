// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/mul_conv_fusion.hpp"
#include "itt.hpp"

#include <memory>
#include <vector>

#include <ngraph/ngraph.hpp>
#include <ngraph/pattern/matcher.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/opsets/opset8.hpp>

#include <transformations/utils/utils.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::MultiplyConvolutionFusion, "MultiplyConvolutionFusion", 0);

ngraph::pass::MultiplyConvolutionFusion::MultiplyConvolutionFusion() {
    MATCHER_SCOPE(MultiplyConvolutionFusion);
    auto input_pattern = pattern::any_input();
    auto mul_const_pattern = ngraph::pattern::wrap_type<opset8::Constant>();
    auto mul_pattern = ngraph::pattern::wrap_type<opset8::Multiply>({input_pattern, mul_const_pattern}, pattern::consumers_count(1));
    auto weights_pattern = ngraph::pattern::any_input(pattern::has_static_shape());
    auto conv_pattern = ngraph::pattern::wrap_type<opset8::Convolution, opset8::ConvolutionBackpropData,
                                                   opset8::GroupConvolution, opset8::GroupConvolutionBackpropData>({mul_pattern, weights_pattern});

    matcher_pass_callback callback = [=](pattern::Matcher & m) -> bool {
        const auto& pattern_to_output = m.get_pattern_value_map();

        const auto& weights = pattern_to_output.at(weights_pattern);
        const auto& mul_const = pattern_to_output.at(mul_const_pattern);

        const auto& weights_shape = weights.get_shape();
        const auto& mul_const_shape = mul_const.get_shape();
        if (op::util::check_for_broadcast(weights_shape, mul_const_shape)) {
            return false;
        }

        auto weights_multiply = std::make_shared<opset8::Multiply>(weights, mul_const);
        std::shared_ptr<Node> new_weights = get_constant_from_source(weights_multiply);
        if (!new_weights)
            new_weights = weights_multiply;

        const auto& input = pattern_to_output.at(input_pattern);
        const auto& conv = pattern_to_output.at(conv_pattern).get_node_shared_ptr();

        auto new_conv = conv->clone_with_new_inputs({input, new_weights});
        new_conv->set_friendly_name(conv->get_friendly_name());
        copy_runtime_info({conv, pattern_to_output.at(mul_pattern).get_node_shared_ptr()},
                          {new_weights, new_conv});
        replace_node(conv, new_conv);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv_pattern, matcher_name);
    register_matcher(m, callback);
}
