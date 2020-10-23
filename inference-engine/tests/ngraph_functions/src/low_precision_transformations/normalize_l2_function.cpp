// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/normalize_l2_function.hpp"

#include <ngraph_ops/type_relaxed.hpp>
#include <ngraph/opsets/opset1.hpp>
#include "ngraph_functions/subgraph_builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> NormalizeL2Function::getOriginal(
    const ngraph::element::Type precision,
    const std::pair<ngraph::Shape, ngraph::Shape>& shapes,
    const ngraph::element::Type precisionOnActivation,
    const std::vector<uint64_t>& axes,
    const bool fuseMultiply,
    const bool shift) {
    const float low = precisionOnActivation == ngraph::element::u8 ? (0.f + (shift ? 10.f : 0.f)) : (-128.f + (shift ? 10.f : 0.f));
    const float high = precisionOnActivation == ngraph::element::u8 ? 255.f : 127.f;
    const float inputScale = 10.f;
    const float outputScale = 20.f;


    const auto paramNode = std::make_shared<ngraph::opset1::Parameter>(precision, shapes.first);
    paramNode->set_friendly_name("input");

    const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
        paramNode->output(0), precision, 256, shapes.second,
        { low / inputScale }, { high / inputScale }, { low / outputScale }, { high / outputScale });

    fakeQuantize->set_friendly_name("fakeQuantize");

    const auto axesNode = std::make_shared<ngraph::op::Constant>(ngraph::element::u64, ngraph::Shape{ axes.size() }, axes);
    axesNode->set_friendly_name("axes");
    const auto normalizeL2 = std::make_shared<ngraph::opset1::NormalizeL2>(fakeQuantize->output(0), axesNode, 1e-6, ngraph::op::EpsMode::ADD);
    normalizeL2->set_friendly_name("normalizeL2");

    ngraph::ResultVector results;
    if (fuseMultiply) {
        const auto multiplyConst = std::make_shared<ngraph::op::Constant>(
            precision, ngraph::Shape{ shapes.first[0], shapes.first[1], 1ul, 1ul }, std::vector<float>{ 2.f });
        multiplyConst->set_friendly_name("multiplyConst");
        const auto multiply = std::make_shared<ngraph::opset1::Multiply>(normalizeL2->output(0), multiplyConst);
        multiply->set_friendly_name("output");

        results = { std::make_shared<ngraph::opset1::Result>(multiply) };
    } else {
        normalizeL2->set_friendly_name("output");
        results = { std::make_shared<ngraph::opset1::Result>(normalizeL2) };
    }

    const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ paramNode }, "NormalizeL2Transformation");
    return function;
}

std::shared_ptr<ngraph::Function> NormalizeL2Function::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& shape,
    const ngraph::op::EpsMode& epsMode,
    const NormalizeL2ActualValues& actualValues) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(actualValues.precision, shape);
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
                precision, Shape({ 1, actualValues.mutliplyValues.size(), 1, 1 }), actualValues.mutliplyValues));
        parent = multiply;
    }

    const auto axesNode = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ actualValues.axes.size() }, actualValues.axes);
    const auto normalizeL2 = std::make_shared<ngraph::opset1::NormalizeL2>(parent, axesNode, 1e-6, epsMode);
    normalizeL2->set_friendly_name("output");

    ngraph::ResultVector results = { std::make_shared<ngraph::opset1::Result>(normalizeL2) };
    const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "NormalizeL2Transformation");
    return function;
}

std::shared_ptr<ngraph::Function> NormalizeL2Function::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& shape,
    const ngraph::op::EpsMode& epsMode,
    const NormalizeL2ExpectedValues& expectedValues) {
    const auto input = std::make_shared<ngraph::opset1::Parameter>(expectedValues.precision, shape);
    std::shared_ptr<ngraph::Node> parent = input;

    if (!expectedValues.subtractValues.empty()) {
        const std::shared_ptr<ngraph::Node> convert = std::make_shared<ngraph::opset1::Convert>(parent, precision);
        parent = convert;

        const std::shared_ptr<ngraph::Node> subtract = std::make_shared<op::TypeRelaxed<ngraph::opset1::Subtract>>(
            std::vector<ngraph::element::Type>{ element::f32, element::f32 }, std::vector<ngraph::element::Type>{element::f32},
            ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(std::make_shared<ngraph::opset1::Constant>(
                precision,
                Shape({ expectedValues.subtractValues.size() }),
                expectedValues.subtractValues), element::f32).get());
        parent = subtract;
    }

    const auto axesNode = std::make_shared<ngraph::op::Constant>(ngraph::element::i64, ngraph::Shape{ expectedValues.axes.size() }, expectedValues.axes);
    const auto normalizeL2 = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::NormalizeL2>>(
        std::vector<ngraph::element::Type>{ element::f32, element::f32 }, std::vector<ngraph::element::Type>{element::f32},
        ngraph::op::TemporaryReplaceOutputType(parent, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(axesNode, element::f32).get(),
        1e-6,
        epsMode);
    std::shared_ptr<ngraph::Node> output = normalizeL2;

    if (!expectedValues.mutliplyValues.empty()) {
        const std::shared_ptr<ngraph::Node> multiply = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::Multiply>>(
            std::vector<ngraph::element::Type>{ element::f32, element::f32 }, std::vector<ngraph::element::Type>{element::f32},
            ngraph::op::TemporaryReplaceOutputType(output, element::f32).get(),
            ngraph::op::TemporaryReplaceOutputType(std::make_shared<ngraph::opset1::Constant>(
                precision, Shape({ 1, expectedValues.mutliplyValues.size(), 1, 1 }), expectedValues.mutliplyValues), element::f32).get());
        output = multiply;
    }
    output->set_friendly_name("output");

    ngraph::ResultVector results = { std::make_shared<ngraph::opset1::Result>(output) };
    const auto function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input }, "NormalizeL2Transformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
