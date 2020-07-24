// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/depth_to_space_function.hpp"

#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> DepthToSpaceFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
    const size_t blockSize) {
    const float low = 0.f;
    const float high = 255.f;
    const float inputScale = 10.f;
    const float outputScale = 20.f;

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape);

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        input, precision, 256, { 1, 1, 1, 1 },
        { low / inputScale }, { high / inputScale }, { low / outputScale }, { high / outputScale });

    auto d2s = std::make_shared<ngraph::opset1::DepthToSpace>(fakeQuantize, mode, blockSize);

    ngraph::ResultVector results = { std::make_shared<ngraph::opset1::Result>(d2s) };

    const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "DepthToSpaceTransformation");
    return function;
}

std::shared_ptr<ngraph::Function> DepthToSpaceFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
    const size_t blockSize,
    const DepthToSpaceActualValues& actualValues) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(actualValues.precision, inputShape);
    std::shared_ptr<ngraph::Node> parent = input;

    const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, precision);
    parent = convert;

    if (!actualValues.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared< ngraph::opset1::Subtract >(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ actualValues.subtractValues.size() }), actualValues.subtractValues));
        parent = subtract;
    }

    if (!actualValues.mutliplyValues.empty()) {
        const std::shared_ptr<ngraph::Node> multiply = std::make_shared< ngraph::opset1::Multiply >(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ actualValues.mutliplyValues.size() }), actualValues.mutliplyValues));
        parent = multiply;
    }

    auto d2s = std::make_shared<ngraph::opset1::DepthToSpace>(parent, mode, blockSize);

    ngraph::ResultVector results = { std::make_shared<ngraph::opset1::Result>(d2s) };

    const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "DepthToSpaceTransformation");
    return function;
}

std::shared_ptr<ngraph::Function> DepthToSpaceFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
    const size_t blockSize,
    const DepthToSpaceExpectedValues& expectedValues) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(expectedValues.precision, inputShape);

    auto d2s = std::make_shared<ngraph::opset1::DepthToSpace>(input, mode, blockSize);

    std::shared_ptr<ngraph::Node> parent = d2s;

    const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, precision);
    parent = convert;

    if (!expectedValues.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> subtract = std::make_shared< ngraph::opset1::Subtract >(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ expectedValues.subtractValues.size() }), expectedValues.subtractValues));
        parent = subtract;
    }

    if (!expectedValues.mutliplyValues.empty()) {
        const std::shared_ptr<ngraph::Node> multiply = std::make_shared< ngraph::opset1::Multiply >(
            parent,
            std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ expectedValues.mutliplyValues.size() }), expectedValues.mutliplyValues));
        parent = multiply;
    }

    ngraph::ResultVector results = { std::make_shared<ngraph::opset1::Result>(parent) };

    const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "DepthToSpaceTransformation");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
