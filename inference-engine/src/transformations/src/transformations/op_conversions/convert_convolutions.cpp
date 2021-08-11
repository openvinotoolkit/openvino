// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/convert_convolutions.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset1.hpp>
#include <ngraph/rt_info.hpp>

#include <ngraph_ops/convolution_ie.hpp>
#include <ngraph_ops/deconvolution_ie.hpp>

#include <ngraph/pattern/op/wrap_type.hpp>

NGRAPH_RTTI_DEFINITION(ov::pass::ConvertConvolutions, "ConvertConvolutions", 0);
NGRAPH_RTTI_DEFINITION(ov::pass::ConvertConvolution, "ConvertConvolution", 0);

ov::pass::ConvertConvolution::ConvertConvolution() {
    MATCHER_SCOPE(ConvertConvolution);
    auto conv = ov::pattern::wrap_type<opset1::Convolution>();

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto conv = std::dynamic_pointer_cast<ov::opset1::Convolution> (m.get_match_root());
        if (!conv) {
            return false;
        }

        auto conv_ie = std::make_shared<ov::op::ConvolutionIE>(conv->input_value(0),
                                                                   conv->input_value(1),
                                                                   conv->get_strides(),
                                                                   conv->get_dilations(),
                                                                   conv->get_pads_begin(),
                                                                   conv->get_pads_end(),
                                                                   conv->get_output_element_type(0),
                                                                   1 /* groups */,
                                                                   conv->get_auto_pad());
        ov::copy_runtime_info(conv, conv_ie);
        conv_ie->set_friendly_name(conv->get_friendly_name());
        ov::replace_node(conv, conv_ie);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(conv, matcher_name);
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ov::pass::ConvertGroupConvolution, "ConvertGroupConvolution", 0);

ov::pass::ConvertGroupConvolution::ConvertGroupConvolution() {
    MATCHER_SCOPE(ConvertGroupConvolution);
    auto gconv = ov::pattern::wrap_type<opset1::GroupConvolution>();

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto gconv = std::dynamic_pointer_cast<ov::opset1::GroupConvolution> (m.get_match_root());
        if (!gconv) {
            return false;
        }
        size_t group = gconv->input_value(1).get_shape()[0];

        // Merge weights layout GOIYX to (G*O)IYX
        auto shape = gconv->input_value(1).get_shape();
        Shape reshape_shape{static_cast<size_t>(shape[0] * shape[1])};
        for (size_t i = 2; i < shape.size(); ++i) {
            reshape_shape.push_back(shape[i]);
        }
        Output<Node> weights;
        auto w_input = gconv->input_value(1).get_node_shared_ptr();
        if (std::dynamic_pointer_cast<opset1::Reshape>(w_input) && w_input->input_value(0).get_shape() == reshape_shape) {
            weights = w_input->input_value(0);
        } else {
            weights = std::make_shared<ov::opset1::Reshape>(gconv->input_value(1),
                                                                op::Constant::create(element::i64, Shape{reshape_shape.size()}, reshape_shape), true);
            ov::copy_runtime_info(gconv, weights.get_node_shared_ptr());
        }
        auto conv_ie = std::make_shared<ov::op::ConvolutionIE>(gconv->input_value(0),
                                                                   weights,
                                                                   gconv->get_strides(),
                                                                   gconv->get_dilations(),
                                                                   gconv->get_pads_begin(),
                                                                   gconv->get_pads_end(),
                                                                   gconv->get_output_element_type(0),
                                                                   group,
                                                                   gconv->get_auto_pad());
        conv_ie->set_friendly_name(gconv->get_friendly_name());
        ov::copy_runtime_info(gconv, conv_ie);
        ov::replace_node(gconv, conv_ie);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(gconv, matcher_name);
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ov::pass::ConvertDeconvolution, "ConvertDeconvolution", 0);

ov::pass::ConvertDeconvolution::ConvertDeconvolution() {
    MATCHER_SCOPE(ConvertDeconvolution);
    auto conv = ov::pattern::wrap_type<opset1::ConvolutionBackpropData>();

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto deconv = std::dynamic_pointer_cast<ov::opset1::ConvolutionBackpropData> (m.get_match_root());
        if (!deconv) {
            return false;
        }

        auto deconv_ie = std::make_shared<ov::op::DeconvolutionIE>(deconv->input_value(0),
                                                                       deconv->input_value(1),
                                                                       deconv->get_strides(),
                                                                       deconv->get_dilations(),
                                                                       deconv->get_pads_begin(),
                                                                       deconv->get_pads_end(),
                                                                       deconv->get_output_element_type(0),
                                                                       1 /* groups */,
                                                                       deconv->get_auto_pad(),
                                                                       deconv->get_output_padding(),
                                                                       (deconv->inputs().size() == 3 ? deconv->input_value(2).get_node_shared_ptr()
                                                                                                     : nullptr));
        deconv_ie->set_friendly_name(deconv->get_friendly_name());
        ov::copy_runtime_info(deconv, deconv_ie);
        ov::replace_node(deconv, deconv_ie);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(conv, matcher_name);
    this->register_matcher(m, callback);
}

NGRAPH_RTTI_DEFINITION(ov::pass::ConvertGroupDeconvolution, "ConvertGroupDeconvolution", 0);

ov::pass::ConvertGroupDeconvolution::ConvertGroupDeconvolution() {
    MATCHER_SCOPE(ConvertGroupDeconvolution);
    auto gconv = ov::pattern::wrap_type<opset1::GroupConvolutionBackpropData>();

    ov::matcher_pass_callback callback = [](pattern::Matcher& m) {
        auto gconv = std::dynamic_pointer_cast<ov::opset1::GroupConvolutionBackpropData> (m.get_match_root());
        if (!gconv) {
            return false;
        }
        size_t group = gconv->input_value(1).get_shape()[0];

        // Merge weights layout GIOYX to I(G*O)YX
        auto input_shape = gconv->input_value(0).get_shape();
        auto weights_shape = gconv->input_value(1).get_shape();
        std::vector<int64_t> reshape_shape{static_cast<int64_t>(weights_shape[1]),
                                           static_cast<int64_t>(weights_shape[2] * group)};
        for (size_t i = 3; i < weights_shape.size(); ++i) {
            reshape_shape.push_back(weights_shape[i]);
        }

        auto reshape = std::make_shared<ov::opset1::Reshape>(gconv->input_value(1),
                                                                 op::Constant::create(element::i64, Shape{reshape_shape.size()}, reshape_shape), true);
        auto conv_ie = std::make_shared<ov::op::DeconvolutionIE>(gconv->input_value(0),
                                                                     reshape,
                                                                     gconv->get_strides(),
                                                                     gconv->get_dilations(),
                                                                     gconv->get_pads_begin(),
                                                                     gconv->get_pads_end(),
                                                                     gconv->get_output_element_type(0),
                                                                     group,
                                                                     gconv->get_auto_pad(),
                                                                     gconv->get_output_padding(),
                                                                     (gconv->inputs().size() == 3 ? gconv->input_value(2).get_node_shared_ptr()
                                                                                                  : nullptr));
        conv_ie->set_friendly_name(gconv->get_friendly_name());
        ov::copy_runtime_info(gconv, conv_ie);
        ov::replace_node(gconv, conv_ie);
        return true;
    };

    auto m = std::make_shared<ov::pattern::Matcher>(gconv, matcher_name);
    this->register_matcher(m, callback);
}
