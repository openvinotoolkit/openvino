// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/subtract.hpp"
#include "low_precision/network_helper.hpp"

#include <ngraph/opsets/opset1.hpp>
#include "ov_lpt_models/common/builders.hpp"
#include "ov_models/subgraph_builders.hpp"

using namespace ov::pass::low_precision;

namespace ngraph {
namespace builder {
namespace subgraph {
    std::shared_ptr<ngraph::Function> SubtractFunction::getOriginal(
        const ngraph::element::Type precision,
        const ngraph::PartialShape& inputShape) {
        const float k = 50.f;

        const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);
        const auto fakeQuantizeOnActivations = ngraph::builder::makeFakeQuantize(
            input, precision, 256ul, { 1ul },
            { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });

        const size_t channelsValue = inputShape[1].get_length();
        const auto weights = ngraph::opset1::Constant::create(
            precision,
            ngraph::Shape{ channelsValue, channelsValue, 1, 1 },
            std::vector<float>(channelsValue * channelsValue, 1));

        const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
            fakeQuantizeOnActivations == nullptr ? input : fakeQuantizeOnActivations,
            ngraph::builder::makeFakeQuantize(weights, precision, 256ul, { 1ul }, { -128.f / k }, { 127.f / k }, { -128.f / k }, { 127.f / k }),
            ngraph::Strides{ 1, 1 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::CoordinateDiff{ 0, 0 },
            ngraph::Strides{ 1, 1 });

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(convolution) };
        std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
            results,
            ngraph::ParameterVector{ input },
            "SubtractTransformation");

        return function;
    }

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
