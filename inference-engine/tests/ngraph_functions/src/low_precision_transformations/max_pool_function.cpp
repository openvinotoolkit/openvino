// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/max_pool_function.hpp"

#include <ngraph/opsets/opset1.hpp>
#include <ngraph_ops/type_relaxed.hpp>
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MaxPoolFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ActualValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(
        params.updatePrecisions ? values.lowPrecision : precision,
        ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, precision);
    parent = convert;

    if (!values.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::opset1::Subtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(precision, Shape({ values.subtractValues.size() }), values.subtractValues));
        parent = subtract;
    }

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape({ values.mutliplyValues.size() }), values.mutliplyValues));
    parent = multiply;

    const std::shared_ptr<ngraph::Node> maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        parent,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        op::RoundingType::FLOOR);

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(maxPool) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MaxPoolTransformation");
}

std::shared_ptr<ngraph::Function> MaxPoolFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::pass::low_precision::LayerTransformation::Params& params,
    const ExpectedValues& values) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(
        params.updatePrecisions ? values.activationPrecision : precision,
        ngraph::Shape(inputShape));
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> maxPool = std::make_shared<ngraph::opset1::MaxPool>(
        parent,
        Strides{ 1, 1 },
        Shape{ 1, 1 },
        Shape{ 0, 0 },
        Shape{ 2, 2 },
        op::RoundingType::FLOOR);
    parent = maxPool;

    const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, precision);
    parent = convert;

    if (!values.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<ngraph::opset1::Subtract>(
            parent,
            std::make_shared<ngraph::opset1::Constant>(precision, Shape({ values.subtractValues.size() }), values.subtractValues));
        parent = subtract;
    }

    const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::opset1::Multiply>(
        parent,
        std::make_shared<ngraph::opset1::Constant>(precision, Shape({ values.mutliplyValues.size() }), values.mutliplyValues));

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(multiply) };
    return std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "MaxPoolTransformation");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
