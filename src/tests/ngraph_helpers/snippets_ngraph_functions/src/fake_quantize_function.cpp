// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fake_quantize_function.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/snippets_isa.hpp>
#include <snippets/op/subgraph.hpp>
#include "ngraph_functions/builders.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
std::shared_ptr<ngraph::op::FakeQuantize> getFakeQuantize(
    const std::shared_ptr<Node>& parent,
    const ngraph::Shape& inputShape,
    const element::Type inputType,
    const std::vector<ngraph::Shape>& fakeQuantizeShapes,
    const float zeroPoint) {
    auto generate = [](const ov::element::Type precision,
                       const ngraph::Shape& shape,
                       const float initialValue,
                       const std::string& name) {
        const auto size = ngraph::shape_size(shape);
        std::vector<float> values(size);
        for (auto i = 0; i < size; ++i) {
            values[i] = static_cast<float>(initialValue + i);
        }
        auto constant = std::make_shared<ngraph::opset1::Constant>(precision, shape, values);
        constant->set_friendly_name(name);
        return constant;
    };

    const auto fakeQuantize = std::make_shared<ngraph::opset1::FakeQuantize>(
        parent,
        generate(inputType, fakeQuantizeShapes[0], zeroPoint, "inputLow"),
        generate(inputType, fakeQuantizeShapes[1], 20.f, "inputHigh"),
        generate(inputType, fakeQuantizeShapes[2], zeroPoint, "outputLow"),
        generate(inputType, fakeQuantizeShapes[3], 20.f, "outputHigh"),
        256ul);
    fakeQuantize->set_friendly_name("fakeQuantize");

    return fakeQuantize;
}
} // namespace

std::shared_ptr<ov::Model> FakeQuantizeFunction::get(
    const ngraph::Shape& inputShape,
    const element::Type inputType,
    const std::vector<ngraph::Shape>& fakeQuantizeShapes,
    const float zeroPoint) {
    assert(fakeQuantizeShapes.size() == 4ul);

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    const auto convert1 = std::make_shared<ngraph::opset1::Convert>(parameter, ov::element::u8);
    convert1->set_friendly_name("convert1");

    const auto relu1 = std::make_shared<ngraph::opset1::Relu>(convert1);
    relu1->set_friendly_name("relu1");

    const auto convert2 = std::make_shared<ngraph::opset1::Convert>(relu1, ov::element::f32);
    convert2->set_friendly_name("convert2");

    const auto slope2 = std::make_shared<ngraph::opset1::Constant>(ov::element::f32, ov::Shape{}, std::vector<float>{-1.f});
    const auto relu2 = std::make_shared<ngraph::opset1::PRelu>(convert2, slope2);
    relu2->set_friendly_name("relu2");

    const auto fakeQuantize = getFakeQuantize(relu2, inputShape, inputType, fakeQuantizeShapes, zeroPoint);
    fakeQuantize->set_friendly_name("fakeQuantize");

    const auto relu3 = std::make_shared<ngraph::opset1::Relu>(fakeQuantize);
    relu3->set_friendly_name("relu3");

    const auto result = std::make_shared<ngraph::opset1::Result>(relu3);
    result->set_friendly_name("result");

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{parameter}, "FakeQuantizeFunction");
}

std::shared_ptr<ov::Model> FakeQuantizeFunction::getSubgraphWithFakeQuantize(
    const ngraph::Shape& inputShape,
    const element::Type inputType,
    const std::vector<ngraph::Shape>& fakeQuantizeShapes,
    const float zeroPoint) {
    assert(fakeQuantizeShapes.size() == 4ul);

    auto getSubgraphBody = [](
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ngraph::Shape>& fakeQuantizeShapes,
        const float zeroPoint) {
        const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
        parameter->set_friendly_name("parameter");

        const auto fakeQuantize = getFakeQuantize(parameter, inputShape, inputType, fakeQuantizeShapes, zeroPoint);

        const auto result = std::make_shared<ngraph::opset1::Result>(fakeQuantize);
        result->set_friendly_name("result");

        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{parameter}, "SubgraphWithFakeQuantizeBody");
    };

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    const auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(
        ngraph::OutputVector {parameter},
        getSubgraphBody(inputShape, inputType, fakeQuantizeShapes, zeroPoint));
    subgraph->set_friendly_name("subgraph");

    const auto result = std::make_shared<ngraph::opset1::Result>(subgraph);
    result->set_friendly_name("result");

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{parameter}, "SubgraphWithFakeQuantize");
}

std::shared_ptr<ov::Model> FakeQuantizeFunction::getSubgraphWithDecomposedFakeQuantize(
    const ngraph::Shape& inputShape,
    const element::Type inputType,
    const std::vector<ngraph::Shape>& fakeQuantizeShapes,
    const float zeroPoint) {
    assert(fakeQuantizeShapes.size() == 4ul);

    auto getSubgraphBody = [](
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ngraph::Shape>& fakeQuantizeShapes,
        const float zeroPoint) {
        const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
        parameter->set_friendly_name("parameter");

        const auto maximum = std::make_shared<ngraph::opset1::Maximum>(
            parameter,
            std::make_shared<ngraph::opset1::Constant>(element::f32, Shape{}, std::vector<float>{1.f}));
        maximum->set_friendly_name("inputLow");

        const auto minimum = std::make_shared<ngraph::opset1::Minimum>(
            maximum,
            std::make_shared<ngraph::opset1::Constant>(element::f32, Shape{}, std::vector<float>{20.f}));
        minimum->set_friendly_name("inputHigh");

        const auto multiply = std::make_shared<ngraph::opset1::Multiply>(
            minimum,
            std::make_shared<ngraph::opset1::Constant>(element::f32, Shape{}, std::vector<float>{13.4211f}));
        multiply->set_friendly_name("multiply");

        const auto subtract = std::make_shared<ngraph::opset1::Subtract>(
            multiply,
            std::make_shared<ngraph::opset1::Constant>(element::f32, Shape{}, std::vector<float>{13.4211f}));
        subtract->set_friendly_name("subtract");

        const auto round = std::make_shared<ngraph::opset5::Round>(subtract, ngraph::opset5::Round::RoundMode::HALF_TO_EVEN);
        round->set_friendly_name("round");

        const auto devide = std::make_shared<ngraph::opset1::Multiply>(
            round,
            std::make_shared<ngraph::opset1::Constant>(element::f32, Shape{}, std::vector<float>{0.0745098f}));
        devide->set_friendly_name("devide");

        const auto add = std::make_shared<ngraph::opset1::Add>(
            devide,
            std::make_shared<ngraph::opset1::Constant>(element::f32, Shape{}, std::vector<float>{1.f}));
        add->set_friendly_name("add");

        const auto result = std::make_shared<ngraph::opset1::Result>(add);
        result->set_friendly_name("result");

        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{parameter}, "SubgraphWithDecomposedFakeQuantizeBody");
    };

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    const auto subgraph = std::make_shared<ngraph::snippets::op::Subgraph>(
        ngraph::OutputVector {parameter},
        getSubgraphBody(inputShape, inputType, fakeQuantizeShapes, zeroPoint));
    subgraph->set_friendly_name("subgraph");

    const auto result = std::make_shared<ngraph::opset1::Result>(subgraph);
    result->set_friendly_name("result");

    return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{parameter}, "SubgraphWithDecomposedFakeQuantize");
}

}  // namespace snippets
}  // namespace test
}  // namespace ov
