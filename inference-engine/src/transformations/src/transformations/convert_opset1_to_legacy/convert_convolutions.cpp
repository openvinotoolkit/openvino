// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/convert_opset1_to_legacy/convert_convolutions.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph_ops/deconvolution_ie.hpp>

void ngraph::pass::ConvertConvolutions::convert_convolution() {
    auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3, 12, 12});
    auto weights = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3, 1, 1});
    auto conv = std::make_shared<ngraph::opset1::Convolution>(data,
                                                              weights,
                                                              Strides{1, 1},
                                                              CoordinateDiff{0, 0},
                                                              CoordinateDiff{0, 0},
                                                              Strides{1, 1});

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<ngraph::opset1::Convolution> (m.get_match_root());
        if (!conv) {
            return false;
        }

        auto conv_ie = std::make_shared<ngraph::op::ConvolutionIE>(conv->input_value(0),
                                                                   conv->input_value(1),
                                                                   conv->get_strides(),
                                                                   conv->get_dilations(),
                                                                   conv->get_pads_begin(),
                                                                   conv->get_pads_end(),
                                                                   1 /* groups */,
                                                                   conv->get_auto_pad());
        ngraph::copy_runtime_info(conv, conv_ie);
        conv_ie->set_friendly_name(conv->get_friendly_name());
        ngraph::replace_node(conv, conv_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "ConvertConvolution");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

void ngraph::pass::ConvertConvolutions::convert_group_convolution() {
    auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3, 12, 12});
    auto weights = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 1, 1, 1, 1});
    auto gconv = std::make_shared<ngraph::opset1::GroupConvolution>(data,
                                                                    weights,
                                                                    Strides{1, 1},
                                                                    CoordinateDiff{0, 0},
                                                                    CoordinateDiff{0, 0},
                                                                    Strides{1, 1});

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto gconv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution> (m.get_match_root());
        if (!gconv) {
            return false;
        }
        size_t group = gconv->input_value(1).get_shape()[0];

        // Merge weights layout GOIYX to (G*O)IYX
        auto shape = gconv->input_value(1).get_shape();
        std::vector<int64_t> reshape_shape{-1};
        for (size_t i = 2; i < shape.size(); ++i) {
            reshape_shape.push_back(shape[i]);
        }
        Output<Node> weights;
        auto w_input = gconv->input_value(1).get_node_shared_ptr();
        if (std::dynamic_pointer_cast<opset1::Reshape>(w_input) && w_input->input_value(0).get_shape().size() == w_input->get_output_shape(0).size() - 1) {
            weights = w_input->input_value(0);
        } else {
            weights = std::make_shared<ngraph::opset1::Reshape>(gconv->input_value(1),
                                                                op::Constant::create(element::i64, Shape{reshape_shape.size()}, reshape_shape), true);
        }
        auto conv_ie = std::make_shared<ngraph::op::ConvolutionIE>(gconv->input_value(0),
                                                                   weights,
                                                                   gconv->get_strides(),
                                                                   gconv->get_dilations(),
                                                                   gconv->get_pads_begin(),
                                                                   gconv->get_pads_end(),
                                                                   group,
                                                                   gconv->get_auto_pad());
        conv_ie->set_friendly_name(gconv->get_friendly_name());
        ngraph::copy_runtime_info(gconv, conv_ie);
        ngraph::replace_node(gconv, conv_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gconv, "ConvertGroupConvolution");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

void ngraph::pass::ConvertConvolutions::convert_convolution_backprop_data() {
    auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3, 12, 12});
    auto weights = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3, 1, 1});
    auto conv = std::make_shared<ngraph::opset1::ConvolutionBackpropData>(data,
                                                                          weights,
                                                                          Strides{1, 1},
                                                                          CoordinateDiff{0, 0},
                                                                          CoordinateDiff{0, 0},
                                                                          Strides{1, 1});

    auto output_shape = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
    auto conv2 = std::make_shared<ngraph::opset1::ConvolutionBackpropData>(data,
                                                                           weights,
                                                                           output_shape,
                                                                           Strides{1, 1},
                                                                           CoordinateDiff{0, 0},
                                                                           CoordinateDiff{0, 0},
                                                                           Strides{1, 1});

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto deconv = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData> (m.get_match_root());
        if (!deconv) {
            return false;
        }

        auto deconv_ie = std::make_shared<ngraph::op::DeconvolutionIE>(deconv->input_value(0),
                                                                       deconv->input_value(1),
                                                                       deconv->get_strides(),
                                                                       deconv->get_pads_begin(),
                                                                       deconv->get_pads_end(),
                                                                       deconv->get_dilations(),
                                                                       deconv->output(0).get_shape(),
                                                                       1 /* groups */,
                                                                       deconv->get_auto_pad());
        deconv_ie->set_friendly_name(deconv->get_friendly_name());
        ngraph::copy_runtime_info(deconv, deconv_ie);
        ngraph::replace_node(deconv, deconv_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(conv, "ConvertConvolutionBackpropData");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);

    auto m2 = std::make_shared<ngraph::pattern::Matcher>(conv2, "ConvertConvolutionBackpropData2");
    this->add_matcher(m2, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}

void ngraph::pass::ConvertConvolutions::convert_group_convolution_backprop_data() {
    auto data = std::make_shared<pattern::op::Label>(element::f32, Shape{1, 3, 12, 12});
    auto weights = std::make_shared<pattern::op::Label>(element::f32, Shape{3, 1, 1, 1, 1});
    auto gconv = std::make_shared<ngraph::opset1::GroupConvolutionBackpropData>(data,
                                                                                weights,
                                                                                Strides{1, 1},
                                                                                CoordinateDiff{0, 0},
                                                                                CoordinateDiff{0, 0},
                                                                                Strides{1, 1});

    auto output_shape = std::make_shared<pattern::op::Label>(element::i64, Shape{2});
    auto gconv2 = std::make_shared<ngraph::opset1::GroupConvolutionBackpropData>(data,
                                                                                 weights,
                                                                                 output_shape,
                                                                                 Strides{1, 1},
                                                                                 CoordinateDiff{0, 0},
                                                                                 CoordinateDiff{0, 0},
                                                                                 Strides{1, 1});

    ngraph::graph_rewrite_callback callback = [](pattern::Matcher& m) {
        auto gconv = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolutionBackpropData> (m.get_match_root());
        if (!gconv) {
            return false;
        }
        size_t group = gconv->input_value(1).get_shape()[0];

        // Merge weights layout GIOYX to I(G*O)YX
        auto input_shape = gconv->input_value(0).get_shape();
        auto weights_shape = gconv->input_value(1).get_shape();
        std::vector<size_t> reshape_shape{weights_shape[1], weights_shape[2] * group};
        for (size_t i = 3; i < weights_shape.size(); ++i) {
            reshape_shape.push_back(weights_shape[i]);
        }

        auto reshape = std::make_shared<ngraph::opset1::Reshape>(gconv->input_value(1),
                                                                 op::Constant::create(element::i64, Shape{reshape_shape.size()}, reshape_shape), true);
        auto conv_ie = std::make_shared<ngraph::op::DeconvolutionIE>(gconv->input_value(0),
                                                                     reshape,
                                                                     gconv->get_strides(),
                                                                     gconv->get_pads_begin(),
                                                                     gconv->get_pads_end(),
                                                                     gconv->get_dilations(),
                                                                     gconv->output(0).get_shape(),
                                                                     group,
                                                                     gconv->get_auto_pad());
        conv_ie->set_friendly_name(gconv->get_friendly_name());
        ngraph::copy_runtime_info(gconv, conv_ie);
        ngraph::replace_node(gconv, conv_ie);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(gconv, "ConvertGroupConvolutionBackpropData");
    this->add_matcher(m, callback, PassProperty::CHANGE_DYNAMIC_STATE);

    auto m2 = std::make_shared<ngraph::pattern::Matcher>(gconv2, "ConvertGroupConvolutionBackpropData2");
    this->add_matcher(m2, callback, PassProperty::CHANGE_DYNAMIC_STATE);
}
