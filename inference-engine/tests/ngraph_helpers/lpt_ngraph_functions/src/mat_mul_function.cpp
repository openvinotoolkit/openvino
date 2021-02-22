// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "lpt_ngraph_functions/mat_mul_function.hpp"

#include <queue>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"
#include "lpt_ngraph_functions/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ngraph::Function> MatMulFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape inputShape,
    const float low,
    const float high) {
    const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(
        input1, precision, 256ul, { 1ul },
        { low / 4.f }, { high / 4.f }, { low / 4.f }, { high / 4.f });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(
        input2, precision, 256ul, { 1ul },
        { low / 8.f }, { high / 8.f }, { low / 8.f }, { high / 8.f });
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const auto matMul = std::make_shared<ngraph::opset1::MatMul>(
        fakeQuantize1->output(0),
        fakeQuantize2->output(0),
        false,
        false);
    matMul->set_friendly_name("matMul");

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ std::make_shared<ngraph::opset1::Result>(matMul) },
        ngraph::ParameterVector{ input1, input2 },
        "GemmTransformation");

    return function;
}

std::shared_ptr<ngraph::Function> MatMulFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape inputShape1,
    const ngraph::Shape inputShape2,
    const bool transpose1,
    const bool transpose2) {
    const auto paramNode = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape1);
    const std::vector<size_t> constShapes(inputShape1.size(), 1ul);
    const auto fakeQuantizeOnAcitvations = ngraph::builder::makeFakeQuantize(
        paramNode, precision, 256ul, constShapes,
        { 0.f }, { 255.f / 4.f }, { 0.f }, { 255.f / 4.f });
    fakeQuantizeOnAcitvations->set_friendly_name("fakeQuantizeOnAcitvations");

    auto weightsConst = std::make_shared<ngraph::op::Constant>(
        precision,
        inputShape2,
        std::vector<float>({ 1.f }));
    const auto fakeQuantizeOnWeights = ngraph::builder::makeFakeQuantize(
        weightsConst, precision, 256ul, { 1ul, 1ul },
        { -128.f / 8.f }, { 127.f / 8.f }, { -128.f / 8.f }, { 127.f / 8.f });
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    const std::shared_ptr<ngraph::opset1::MatMul> fullyConnected = std::make_shared<ngraph::opset1::MatMul>(
        fakeQuantizeOnAcitvations->output(0),
        fakeQuantizeOnWeights->output(0),
        transpose1,
        transpose2);
    fullyConnected->set_friendly_name("fullyConnected");

    ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(fullyConnected) };
    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        results,
        ngraph::ParameterVector{ paramNode },
        "FullyConnectedTransformation");

    return function;
}


std::shared_ptr<ngraph::Function> MatMulFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape1,
    const FakeQuantizeOnData& fqOnData1,
    const ngraph::Shape& inputShape2,
    const FakeQuantizeOnData& fqOnData2) {
    const std::shared_ptr<ngraph::opset1::Parameter> input1 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape1);
    input1->set_friendly_name("input1");

    const std::shared_ptr<ngraph::opset1::Parameter> input2 = std::make_shared<ngraph::opset1::Parameter>(precision, inputShape2);
    input2->set_friendly_name("input2");

    const std::shared_ptr<ngraph::opset1::MatMul> matMul = std::make_shared<ngraph::opset1::MatMul>(
        makeFakeQuantize(input1, precision, fqOnData1),
        makeFakeQuantize(input2, precision, fqOnData2),
        false,
        false);
    matMul->set_friendly_name("matMul");

    std::shared_ptr<ngraph::opset1::Result> result = std::make_shared<ngraph::opset1::Result>(matMul);

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ result },
        std::vector<std::shared_ptr<ngraph::op::Parameter>> { input1, input2 },
        "MatMulTransformation");
    return function;
}

std::shared_ptr<ngraph::Function> MatMulFunction::getOriginal(
    const element::Type netPrecision,
    const ngraph::Shape& inputShape1,
    const ngraph::element::Type precisionBeforeDequantization1,
    const DequantizationOperations& dequantization1,
    const ngraph::Shape& inputShape2,
    const ngraph::element::Type precisionBeforeDequantization2,
    const DequantizationOperations& dequantization2) {
    if (!dequantization1.convert.empty() && (precisionBeforeDequantization1 == dequantization1.convert.outPrecision)) {
        throw std::runtime_error("unexpected input arguments for branch 1");
    }

    if (!dequantization2.convert.empty() && (precisionBeforeDequantization2 == dequantization2.convert.outPrecision)) {
        throw std::runtime_error("unexpected input arguments for branch 2");
    }

    const std::shared_ptr<ngraph::opset1::Parameter> input1 = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization1, inputShape1);
    input1->set_friendly_name("input1");

    const std::shared_ptr<ngraph::opset1::Parameter> input2 = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization2, inputShape2);
    input2->set_friendly_name("input2");

    DequantizationOperations deqSructure1 = dequantization1;
    deqSructure1.multiply.outPrecision = netPrecision;
    DequantizationOperations deqSructure2 = dequantization2;
    deqSructure2.multiply.outPrecision = netPrecision;

    const std::shared_ptr<ngraph::opset1::MatMul> matMul = std::make_shared<ngraph::opset1::MatMul>(
        makeDequantization(input1, deqSructure1),
        makeDequantization(input2, deqSructure2),
        false,
        false);
    matMul->set_friendly_name("matMul");

    std::shared_ptr<ngraph::opset1::Result> result = std::make_shared<ngraph::opset1::Result>(matMul);

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ result },
        std::vector<std::shared_ptr<ngraph::op::Parameter>> { input1, input2 },
        "MatMulTransformation");
    return function;
}

std::shared_ptr<ngraph::Function> getOriginalWithConstant2(
    const ngraph::element::Type precision) {
    return nullptr;
}

std::shared_ptr<ngraph::Function> MatMulFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const DequantizationOperations& dequantizationOperations,
    const ngraph::Shape& weightsConstShape,
    const std::vector<float>& weightsConstValues,
    const FakeQuantizeOnWeights& fqOnWeights) {
    const std::shared_ptr<ngraph::opset1::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        inputShape);
    input->set_friendly_name("input1");

    auto lastDequantization = makeDequantization(input, dequantizationOperations);

    const std::shared_ptr<ngraph::opset1::Constant> weightsConst = std::make_shared<ngraph::opset1::Constant>(
        precision,
        weightsConstShape,
        weightsConstValues);

    auto fakeQuantize = makeFakeQuantize(weightsConst, precision, fqOnWeights);

    const std::shared_ptr<ngraph::opset1::MatMul> matMul = std::make_shared<ngraph::opset1::MatMul>(
        lastDequantization,
        fakeQuantize,
        false,
        false);
    matMul->set_friendly_name("matMul");
    auto& rtInfo = matMul->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("matMul");

    std::shared_ptr<ngraph::opset1::Result> result = std::make_shared<ngraph::opset1::Result>(matMul);

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ result },
        std::vector<std::shared_ptr<ngraph::op::Parameter>> { input },
        "MatMulTransformation");
    return function;
}

std::shared_ptr<ngraph::Function> MatMulFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape1,
    const ngraph::element::Type precisionBeforeDequantization1,
    const DequantizationOperations& dequantization1,
    const ngraph::Shape& inputShape2,
    const ngraph::element::Type precisionBeforeDequantization2,
    const DequantizationOperations& dequantization2,
    const DequantizationOperations& resultDequantizationOperations) {
    if (!dequantization1.convert.empty() && (precisionBeforeDequantization1 == dequantization1.convert.outPrecision)) {
        throw std::runtime_error("unexpected input arguments for branch 1");
    }

    if (!dequantization2.convert.empty() && (precisionBeforeDequantization2 == dequantization2.convert.outPrecision)) {
        throw std::runtime_error("unexpected input arguments for branch 2");
    }

    const std::shared_ptr<ngraph::opset1::Parameter> input1 = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization1, inputShape1);
    input1->set_friendly_name("input1");

    const std::shared_ptr<ngraph::opset1::Parameter> input2 = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization2, inputShape2);
    input2->set_friendly_name("input2");

    DequantizationOperations deqSructure1 = dequantization1;
    deqSructure1.multiply.outPrecision = precision;
    DequantizationOperations deqSructure2 = dequantization2;
    deqSructure2.multiply.outPrecision = precision;

    auto dequantization1Op = makeDequantization(input1, deqSructure1);
    auto dequantization2Op = makeDequantization(input2, deqSructure2);

    std::shared_ptr<ngraph::opset1::MatMul> matMul = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::MatMul>>(
        std::vector<element::Type>{ element::f32, element::f32 }, std::vector<element::Type>{ element::f32 },
        ngraph::op::TemporaryReplaceOutputType(dequantization1Op, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(dequantization2Op, element::f32).get(),
        false,
        false);

    matMul->set_friendly_name("matMul");
    DequantizationOperations deqSructureAfter = resultDequantizationOperations;
    deqSructureAfter.multiply.outPrecision = precision;
    auto dequantizationAfter = makeDequantization(matMul, deqSructureAfter);
    dequantizationAfter->set_friendly_name("matMul");

    std::shared_ptr<ngraph::opset1::Result> result = std::make_shared<ngraph::opset1::Result>(dequantizationAfter);

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ result },
        std::vector<std::shared_ptr<ngraph::op::Parameter>> { input1, input2 },
        "MatMulTransformation");
    return function;
}

std::shared_ptr<ngraph::Function> MatMulFunction::getReference(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const ngraph::element::Type precisionBeforeDequantization,
    const DequantizationOperations& dequantization,
    const ngraph::element::Type weightsConstPrecision,
    const ngraph::Shape& weightsConstShape,
    const std::vector<float>& weightsConstValues,
    const DequantizationOperations& resultDequantization) {
    const std::shared_ptr<ngraph::opset1::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precisionBeforeDequantization,
        inputShape);
    input->set_friendly_name("input1");

    const std::shared_ptr<ngraph::Node> lastDequantizationBefore = makeDequantization(input, dequantization);

    const std::shared_ptr<ngraph::opset1::Constant> weightsConst = std::make_shared<ngraph::opset1::Constant>(
        weightsConstPrecision,
        weightsConstShape,
        weightsConstValues);

    const std::shared_ptr<ngraph::opset1::MatMul> matMul = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::MatMul>>(
        std::vector<ngraph::element::Type>{ element::f32, element::f32 }, std::vector<ngraph::element::Type>{},
        ngraph::op::TemporaryReplaceOutputType(lastDequantizationBefore, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(weightsConst, element::f32).get(),
        false,
        false);
    matMul->set_friendly_name("matMul");
    auto& rtInfo = matMul->get_rt_info();
    rtInfo["Variant::std::string"] = std::make_shared<VariantWrapper<std::string>>("matMul");
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(matMul, precision);

    const std::shared_ptr<ngraph::Node> lastDequantizationAfter = makeDequantization(matMul, resultDequantization);
    lastDequantizationAfter->set_friendly_name("matMul");

    std::shared_ptr<ngraph::opset1::Result> result = std::make_shared<ngraph::opset1::Result>(lastDequantizationAfter);

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ result },
        std::vector<std::shared_ptr<ngraph::op::Parameter>> { input },
        "MatMulTransformation");
    return function;
}

std::shared_ptr<ngraph::Function> MatMulFunction::getOriginal(
    const ngraph::element::Type precision,
    const ngraph::Shape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const ngraph::Shape& weightsConstShape,
    const std::vector<float>& weightsConstValues,
    const FakeQuantizeOnDataWithConstant& fqOnWeights) {
    const std::shared_ptr<ngraph::opset1::Parameter> input = std::make_shared<ngraph::opset1::Parameter>(
        precision,
        inputShape);
    input->set_friendly_name("input1");

    auto lastDequantization = makeFakeQuantize(input, precision, fqOnData);

    const std::shared_ptr<ngraph::opset1::Constant> weightsConst = std::make_shared<ngraph::opset1::Constant>(
        precision,
        weightsConstShape,
        weightsConstValues);

    auto fakeQuantize = makeFakeQuantize(weightsConst, precision, fqOnWeights);

    const std::shared_ptr<ngraph::opset1::MatMul> matMul = std::make_shared<ngraph::opset1::MatMul>(
        lastDequantization,
        fakeQuantize,
        false,
        true);
    matMul->set_friendly_name("matMul");

    std::shared_ptr<ngraph::opset1::Result> result = std::make_shared<ngraph::opset1::Result>(matMul);

    std::shared_ptr<ngraph::Function> function = std::make_shared<ngraph::Function>(
        ngraph::ResultVector{ result },
        std::vector<std::shared_ptr<ngraph::op::Parameter>> { input },
        "MatMulTransformation");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
