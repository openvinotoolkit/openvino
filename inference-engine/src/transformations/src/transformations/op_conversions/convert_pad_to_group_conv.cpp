// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "itt.hpp"
#include "transformations/op_conversions/convert_pad_to_group_conv.hpp"

#include <memory>
#include <vector>

#include <ngraph/opsets/opset4.hpp>
#include <ngraph/rt_info.hpp>
#include <ngraph/pattern/op/wrap_type.hpp>
#include <ngraph/pattern/op/pattern.hpp>

NGRAPH_RTTI_DEFINITION(ngraph::pass::ConvertPadToGroupConvolution, "ConvertPadToGroupConvolution", 0);

ngraph::pass::ConvertPadToGroupConvolution::ConvertPadToGroupConvolution() {
    MATCHER_SCOPE(ConvertPadToGroupConvolution);
    auto neg = ngraph::pattern::wrap_type<opset4::Pad>(pattern::has_static_dim(1));

    ngraph::matcher_pass_callback callback = [this](pattern::Matcher& m) {
        auto pad = std::dynamic_pointer_cast<ngraph::opset4::Pad> (m.get_match_root());
        if (!pad) {
            return false;
        }

        auto input = pad->input_value(0);
        const auto & channel_dim = input.get_partial_shape()[1].get_length();
        const auto & rank = input.get_partial_shape().rank().get_length();
        if (rank < 4) {
            // We can not create Convolution without spatial dimensions.
            // Also creating Convolution with single spatial dimension won't be effective as
            // we later insert additional Reshape operations.
            return false;
        }

        // Check that Pad has CONSTANT mode and value is equal to 0 if 4th input exists
        if (pad->get_pad_mode() != op::PadMode::CONSTANT) {
            return false;
        }

        if (pad->inputs().size() == 4) {
            if (auto pad_value = std::dynamic_pointer_cast<opset4::Constant>(pad->input_value(3).get_node_shared_ptr())) {
                // pad value is a scalar
                if (pad_value->cast_vector<float>()[0] != 0) {
                    return false;
                }
            }
        }

        // Check that Pad has padding only for spatial dimensions
        const auto & pad_begin = pad->get_pads_begin();
        const auto & pad_end = pad->get_pads_end();

        if (pad_begin.empty() || pad_end.empty()) {
            // pads will be empty if inputs are not constants
            return false;
        }

        // Check that not spatial dimension are not padded
        if (std::any_of(pad_begin.begin(), pad_begin.begin() + 2, [](ptrdiff_t value) { return value != 0; }) ||
            std::any_of(pad_end.begin(), pad_end.begin() + 2, [](ptrdiff_t value) { return value != 0; })) {
            return false;
        }

        // Create fake weights with ones GOIXY
        Shape weights_shape(rank + 1, 1);
        weights_shape[0] = channel_dim; // G dimension
        auto weights = opset4::Constant::create(pad->input(0).get_element_type(), weights_shape, {1});

        // Create GroupConvolution attributes
        Strides stride(rank - 2, 1);
        CoordinateDiff new_pad_begin{pad_begin.begin() + 2, pad_begin.end()};
        CoordinateDiff new_pad_end{pad_end.begin() + 2, pad_end.end()};

        auto conv = std::make_shared<opset4::GroupConvolution>(input, weights, stride, new_pad_begin, new_pad_end, stride);

        conv->set_friendly_name(pad->get_friendly_name());
        ngraph::copy_runtime_info(pad, conv);
        ngraph::replace_node(pad, conv);
        return true;
    };

    auto m = std::make_shared<ngraph::pattern::Matcher>(neg, matcher_name);
    this->register_matcher(m, callback);
}
