// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_can_be_quantized.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/concat.hpp"
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::MarkupCanBeQuantized, "MarkupCanBeQuantized", 0);

bool ngraph::pass::low_precision::MarkupCanBeQuantized::run_on_function(std::shared_ptr<ngraph::Function> f) {
    auto setEmptyInputPrecisions = [](const std::shared_ptr<ngraph::Node>& node) {
        for (auto& input : node->inputs()) {
            auto attribute = low_precision::make_shared_attribute<PrecisionsAttribute>(element::TypeVector{});
            ov::set_precisions(input, attribute);
        }
    };

    for (const auto & node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0 || transformation_callback(node)) {
            continue;
        }

        if (const auto convolution = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node)) {
            if (!ConvolutionTransformation::isQuantizedStatic(convolution)) {
                setEmptyInputPrecisions(convolution);
            }
            continue;
        }
        if (const auto convolutionBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node)) {
            if (!ConvolutionBackpropDataTransformation::isQuantizedStatic(convolutionBackpropData)) {
                setEmptyInputPrecisions(convolutionBackpropData);
            }
            continue;
        }
        if (const auto groupConvolution = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node)) {
            if (!GroupConvolutionTransformation::isQuantizedStatic(groupConvolution)) {
                setEmptyInputPrecisions(groupConvolution);
            }
            continue;
        }
        if (const auto concat = std::dynamic_pointer_cast<ngraph::opset1::Concat>(node)) {
            if (!ConcatTransformation::isQuantizedStatic(concat)) {
                setEmptyInputPrecisions(concat);
            }
            continue;
        }
    }
    return true;
}
