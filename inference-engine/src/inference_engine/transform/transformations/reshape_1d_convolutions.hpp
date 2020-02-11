// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <vector>
#include <memory>

#include <ngraph/pass/graph_rewrite.hpp>

#include "ngraph_ops/convolution_ie.hpp"
#include "utils/utils.hpp"

namespace ngraph {
namespace pass {

class Reshape1DConvolutions;

}  // namespace pass
}  // namespace ngraph

class ngraph::pass::Reshape1DConvolutions: public ngraph::pass::GraphRewrite {
public:
    Reshape1DConvolutions() : GraphRewrite() {
        reshape_convolutions();
    }

private:
    void reshape_convolutions() {
        auto input = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3, 64, 64});
        auto w = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 3, 1, 1});
        auto b = std::make_shared<pattern::op::Label>(element::f32, Shape{3});
        auto conv = std::make_shared<ngraph::op::ConvolutionIE>(input, w,
                                                                Strides{1, 1},
                                                                CoordinateDiff{0, 0},
                                                                CoordinateDiff{0, 0},
                                                                Strides{1, 1}, Shape{});
        auto conv_bias = std::make_shared<ngraph::op::ConvolutionIE>(input, w, b,
                                                                     Strides{1, 1},
                                                                     CoordinateDiff{0, 0},
                                                                     CoordinateDiff{0, 0},
                                                                     Strides{1, 1}, Shape{});

        ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
            auto conv = std::dynamic_pointer_cast<ngraph::op::ConvolutionIE> (m.get_match_root());
            if (!conv || conv->get_shape().size() > 3) {
                return false;
            }
            auto input_shape = conv->input(0).get_shape();
            auto output_shape = conv->output(0).get_shape();

            // Reshape(new_input_shape)->ConvolutionIE(new_output_shape)->Reshape(output_shape)
            auto new_input_shape = input_shape;
            auto new_output_shape = output_shape;

            // Insert H dimension equal to 1
            new_input_shape.insert(new_input_shape.begin() + 2, 1);
            new_output_shape.insert(new_output_shape.begin() + 2, 1);

            auto new_strides = conv->get_strides();
            auto new_dilations = conv->get_dilations();
            auto new_pads_begin = conv->get_pads_begin();
            auto new_pad_end = conv->get_pads_end();

            new_strides.insert(new_strides.begin(), 1);
            new_dilations.insert(new_dilations.begin(), 1);

            new_pads_begin.insert(new_pads_begin.begin(), 0);
            new_pad_end.insert(new_pad_end.begin(), 0);

            auto reshape_begin = op::util::reshapeTo(conv->input_value(0), new_input_shape);
            reshape_begin->set_friendly_name(conv->get_friendly_name() + "/reshape_begin");

            auto create_convolution = [&](const Output<Node> & input) -> std::shared_ptr<Node> {
                Shape new_weights_shape(conv->input_value(1).get_shape());
                new_weights_shape.insert(new_weights_shape.begin() + 2, 1);
                auto weights = op::util::reshapeTo(conv->input_value(1), new_weights_shape);
                if (conv->inputs().size() == 2) {
                    return std::make_shared<op::ConvolutionIE>(input,
                                                               weights,
                                                               new_strides,
                                                               new_pads_begin,
                                                               new_pad_end,
                                                               new_dilations,
                                                               new_output_shape,
                                                               conv->get_group(),
                                                               conv->get_auto_pad());
                } else {
                    return std::make_shared<op::ConvolutionIE>(input,
                                                               weights,
                                                               conv->input_value(2),
                                                               new_strides,
                                                               new_pads_begin,
                                                               new_pad_end,
                                                               new_dilations,
                                                               new_output_shape,
                                                               conv->get_group(),
                                                               conv->get_auto_pad());
                }
            };

            auto new_conv = create_convolution(reshape_begin);
            new_conv->set_friendly_name(conv->get_friendly_name() + "/new");

            auto reshape_end = op::util::reshapeTo(new_conv, output_shape);
            reshape_end->set_friendly_name(conv->get_friendly_name());

            ngraph::replace_node(conv, reshape_end);
            return true;
        };

        auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "Reshape1DConvolutions");
        this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);

        auto m_bias = std::make_shared<ngraph::pattern::Matcher>(conv_bias, "Reshape1DConvolutions");
        this->add_matcher(m_bias, callback, PassProperty::CHANGE_DYNAMIC_STATE);
    }
};
