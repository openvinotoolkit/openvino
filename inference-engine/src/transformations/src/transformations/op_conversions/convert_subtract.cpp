// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/convert_subtract.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertSubtract, "ConvertSubtract", 0);

ngraph::pass::ConvertSubtract::ConvertSubtract() {
    MATCHER_SCOPE(ConvertSubtract);
    auto sub = ngraph::pattern::wrap_type<ngraph::opset1::Subtract>();

    ngraph::matcher_pass_callback callback = [=](pattern::Matcher& m) {
        if (transformation_callback(m.get_match_root())) {
            return false;
        }

        auto sub = std::dynamic_pointer_cast<ngraph::opset1::Subtract>(m.get_match_root());
        if (!sub) {
            return false;
        }

        if (sub->input(0).get_element_type() != sub->input(1).get_element_type()) {
            return false;
        }

        if (sub->input(0).get_element_type() == sub->input(1).get_element_type()) {
            const auto subChildren = sub->output(0).get_target_inputs();
            if (subChildren.size() == 1ul) {
                const std::shared_ptr<Node> child = subChildren.begin()->get_node()->shared_from_this();
                if (child != nullptr) {
                    if (is_type<opset1::Convolution>(child) ||
                        is_type<opset1::ConvolutionBackpropData>(child) ||
                        is_type<opset1::GroupConvolution>(child) ||
                        is_type<opset1::GroupConvolutionBackpropData>(child) ||
                        is_type<opset1::MatMul>(child) ||
                        (is_type<opset1::Reshape>(child) &&
                            (child->output(0).get_target_inputs().size() == 1ul) &&
                            (is_type<opset1::GroupConvolution>(child->output(0).get_target_inputs().begin()->get_node()->shared_from_this()) ||
                             is_type<opset1::GroupConvolutionBackpropData>(child->output(0).get_target_inputs().begin()->get_node()->shared_from_this())))) {
                        const auto input1Type = sub->input(0).get_element_type();
                        const auto input2Type = sub->input(1).get_element_type();
                        if (((input1Type == element::u8) && (input2Type == element::u8)) ||
                            ((input1Type == element::i8) && (input2Type == element::i8))) {
                            // we should not execute transformation by reasons:
                            // 1. LPT asymmetric quantization pattern has to be keep as is
                            // 2. Subtract operation has unsigned/signed integer value which is not safe to multiply by -1
                            return false;
                        }
                    }
                }
            }
        }

        auto neg = std::make_shared<ngraph::opset1::Multiply>(sub->input(1).get_source_output(),
                                                              opset1::Constant::create(sub->get_input_element_type(1), Shape{}, {-1}));

        auto add = std::make_shared<ngraph::opset1::Add>(sub->input(0).get_source_output(), neg);

        add->set_friendly_name(sub->get_friendly_name());
        ngraph::copy_runtime_info(sub, {neg, add});
        ngraph::replace_node(sub, add);

        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(sub, matcher_name);
    this->register_matcher(m, callback);
}
