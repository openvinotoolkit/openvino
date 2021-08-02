// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision/markup_can_be_quantized.hpp"

#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "low_precision/convolution.hpp"
#include "low_precision/convolution_backprop_data.hpp"
#include "low_precision/group_convolution.hpp"
#include "low_precision/network_helper.hpp"
#include "low_precision/rt_info/precisions_attribute.hpp"

using namespace ngraph;

NGRAPH_RTTI_DEFINITION(ngraph::pass::low_precision::MarkupCanBeQuantized, "MarkupCanBeQuantized", 0);

bool ngraph::pass::low_precision::MarkupCanBeQuantized::run_on_function(std::shared_ptr<ngraph::Function> f) {
    auto setEmptyPrecisions = [](const std::shared_ptr<ngraph::Node>& node) {
        for (auto& input : node->inputs()) {
            auto& rt = input.get_rt_info();

            auto attribute = ngraph::pass::low_precision::make_shared_attribute<PrecisionsAttribute>(std::vector<element::Type>());
            auto attributeWrapper = std::make_shared<ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>>(attribute);

            rt.emplace(
                    ngraph::VariantWrapper<std::shared_ptr<PrecisionsAttribute>>::type_info.name,
                    attributeWrapper);
        }
    };

    for (const std::shared_ptr<Node>& node : f->get_ordered_ops()) {
        if (node->get_input_size() == 0 || transformation_callback(node)) {
            continue;
        }

        if (const auto convolution = std::dynamic_pointer_cast<ngraph::opset1::Convolution>(node)) {
            if (!ConvolutionTransformation::isQuantizedStatic(convolution)) {
                setEmptyPrecisions(convolution);
            }
            continue;
        }
        if (const auto convolutionBackpropData = std::dynamic_pointer_cast<ngraph::opset1::ConvolutionBackpropData>(node)) {
            if (!ConvolutionBackpropDataTransformation::isQuantizedStatic(convolutionBackpropData)) {
                setEmptyPrecisions(convolutionBackpropData);
            }
            continue;
        }
        if (const auto groupConvolution = std::dynamic_pointer_cast<ngraph::opset1::GroupConvolution>(node)) {
            if (!GroupConvolutionTransformation::isQuantizedStatic(groupConvolution)) {
                setEmptyPrecisions(groupConvolution);
            }
            continue;
        }
    }
    return true;
}
