// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "fake_quantize_helper.hpp"
#include "common_test_utils/data_utils.hpp"
#include <snippets/snippets_isa.hpp>
#include <snippets/op/subgraph.hpp>
#include "ov_models/builders.hpp"
#include "function_helper.hpp"

namespace ov {
namespace test {
namespace snippets {

namespace {
std::shared_ptr<ngraph::op::FakeQuantize> makeFakeQuantize(
    const Output<Node>& parent,
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

std::shared_ptr<ngraph::opset1::Convolution> makeConvolution(const Output<Node>& parent) {
    const auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 3, 3, 1, 1 }, { 1.f });
    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        parent,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("Convolution");
    return convolution;
}

std::shared_ptr<ngraph::opset1::GroupConvolution> makeGroupConvolution(const Output<Node>& parent) {
    const auto weights = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{ 1, 3, 3, 1, 1 }, { 1.f });
    const auto convolution = std::make_shared<ngraph::opset1::GroupConvolution>(
        parent,
        weights,
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });
    convolution->set_friendly_name("GroupConvolution");
    return convolution;
}

std::shared_ptr<ngraph::opset1::MatMul> makeMatMul(const Output<Node>& parent1, const Output<Node>& parent2) {
    const auto matMul = std::make_shared<ngraph::opset1::MatMul>(parent1, parent2);
    matMul->set_friendly_name("MatMul");
    return matMul;
}

Output<Node> initOperation(std::shared_ptr<Node> operation, const std::vector<Output<Node>>& parents) {
    if (is_type<ngraph::opset1::Convolution>(operation)) {
        assert(parents.size() == 1ul);
        return makeConvolution(parents[0]);
    }

    if (is_type<ngraph::opset1::GroupConvolution>(operation)) {
        assert(parents.size() == 1ul);
        return makeGroupConvolution(parents[0]);
    }

    if (is_type<ngraph::opset1::MatMul>(operation)) {
        assert(parents.size() == 2ul);
        return makeMatMul(parents[0], parents[1]);
    }

    operation->set_argument(0, parents[0]);
    auto elementType = std::string(operation->get_type_name());
    operation->set_friendly_name(elementType);

    return operation;
}

// TODO: workaround while element-wise operations after `Parameter` are not added in Subgraph
std::shared_ptr<Node> getOperations(const std::vector<std::shared_ptr<Node>>& operations, const Output<Node>& parent) {
    Output<Node> currentParent = parent;
    for (auto operation : operations) {
        operation->set_argument(0, currentParent);
        currentParent = operation;
    }
    return currentParent.get_node_shared_ptr();
}

} // namespace

std::shared_ptr<ov::Model> FakeQuantizeFunction::getOperationAndFakeQuantize(
    const ngraph::Shape& inputShape,
    const element::Type inputType,
    const std::vector<ngraph::Shape>& fakeQuantizeShapes,
    const float zeroPoint,
    const std::vector<std::shared_ptr<ngraph::Node>>& prerequisites,
    std::shared_ptr<ngraph::Node> operation) {
    assert(fakeQuantizeShapes.size() == 4ul);

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    auto parent = FunctionHelper::applyPrerequisites(parameter, prerequisites);

    const auto fakeQuantize = makeFakeQuantize(
        operation == nullptr ? parent : initOperation(operation, { parent }),
        inputShape,
        inputType,
        fakeQuantizeShapes,
        zeroPoint);

    fakeQuantize->set_friendly_name("fakeQuantize");

    const auto result = std::make_shared<ngraph::opset1::Result>(fakeQuantize);
    result->set_friendly_name("result");

    auto function = std::make_shared<ngraph::Function>(ngraph::ResultVector{ result }, ParameterVector{ parameter }, "FakeQuantizeFunction");
    function->validate_nodes_and_infer_types();

    return function;
}

std::shared_ptr<ov::Model> FakeQuantizeFunction::getSubgraphWithFakeQuantize(
    const ngraph::Shape& inputShape,
    const element::Type inputType,
    const std::vector<ngraph::Shape>& fakeQuantizeShapes,
    const float zeroPoint,
    const std::vector<std::shared_ptr<ngraph::Node>>& prerequisites,
    const std::vector<std::shared_ptr<Node>>& beforeFakeQuantizeOperations) {
    assert(fakeQuantizeShapes.size() == 4ul);

    auto getSubgraphBody = [](
        const ngraph::Shape& inputShape,
        const element::Type inputType,
        const std::vector<ngraph::Shape>& fakeQuantizeShapes,
        const float zeroPoint,
        const std::vector<std::shared_ptr<Node>>& beforeFakeQuantizeOperations) {
        const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
        parameter->set_friendly_name("parameter");

        const auto fakeQuantize = makeFakeQuantize(
            getOperations(beforeFakeQuantizeOperations, {parameter}), inputShape, inputType, fakeQuantizeShapes, zeroPoint);

        const auto result = std::make_shared<ngraph::opset1::Result>(fakeQuantize);
        result->set_friendly_name("result");

        return std::make_shared<ngraph::Function>(ngraph::ResultVector{result}, ngraph::ParameterVector{parameter}, "SubgraphWithFakeQuantizeBody");
    };

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    auto parent = FunctionHelper::applyPrerequisites(parameter, prerequisites);

    const auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(
        ngraph::OutputVector{ parent },
        getSubgraphBody(inputShape, inputType, fakeQuantizeShapes, zeroPoint, beforeFakeQuantizeOperations));
    subgraph->set_friendly_name("subgraph");

    const auto result = std::make_shared<ngraph::opset1::Result>(subgraph);
    result->set_friendly_name("result");

    auto function = std::make_shared<ngraph::Function>(ngraph::ResultVector{ result }, ParameterVector{ parameter }, "SubgraphWithFakeQuantize");
    function->validate_nodes_and_infer_types();
    return function;
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

        return std::make_shared<ngraph::Function>(
            ngraph::ResultVector{result}, ngraph::ParameterVector{parameter}, "SubgraphWithDecomposedFakeQuantizeBody");
    };

    const auto parameter = std::make_shared<ngraph::opset1::Parameter>(inputType, inputShape);
    parameter->set_friendly_name("parameter");

    const auto subgraph = std::make_shared<ov::snippets::op::Subgraph>(
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
