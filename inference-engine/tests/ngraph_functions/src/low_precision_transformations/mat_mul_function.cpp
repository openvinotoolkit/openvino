// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/mat_mul_function.hpp"

#include <queue>
#include <memory>

#include <ngraph/opsets/opset1.hpp>
#include "ngraph_ops/type_relaxed.hpp"
#include "ngraph_functions/subgraph_builders.hpp"
#include "transformations/low_precision/network_helper.hpp"
#include "ngraph_functions/low_precision_transformations/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

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
    const ngraph::Shape& inputShape1,
    const ngraph::element::Type precisionBeforeDequantization1,
    const DequantizationOperations& dequantization1,
    const ngraph::Shape& inputShape2,
    const ngraph::element::Type precisionBeforeDequantization2,
    const DequantizationOperations& dequantization2) {
    if (!dequantization1.convert.empty() && (precisionBeforeDequantization1 == dequantization1.convert.outPrecision)) {
        THROW_IE_EXCEPTION << "unexpected input arguments for branch 1";
    }

    if (!dequantization2.convert.empty() && (precisionBeforeDequantization2 == dequantization2.convert.outPrecision)) {
        THROW_IE_EXCEPTION << "unexpected input arguments for branch 2";
    }

    const std::shared_ptr<ngraph::opset1::Parameter> input1 = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization1, inputShape1);
    input1->set_friendly_name("input1");

    const std::shared_ptr<ngraph::opset1::Parameter> input2 = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization2, inputShape2);
    input2->set_friendly_name("input2");

    const std::shared_ptr<ngraph::opset1::MatMul> matMul = std::make_shared<ngraph::opset1::MatMul>(
        makeDequantization(input1, dequantization1),
        makeDequantization(input2, dequantization2),
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
        THROW_IE_EXCEPTION << "unexpected input arguments for branch 1";
    }

    if (!dequantization2.convert.empty() && (precisionBeforeDequantization2 == dequantization2.convert.outPrecision)) {
        THROW_IE_EXCEPTION << "unexpected input arguments for branch 2";
    }

    const std::shared_ptr<ngraph::opset1::Parameter> input1 = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization1, inputShape1);
    input1->set_friendly_name("input1");

    const std::shared_ptr<ngraph::opset1::Parameter> input2 = std::make_shared<ngraph::opset1::Parameter>(precisionBeforeDequantization2, inputShape2);
    input2->set_friendly_name("input2");

    auto dequantization1Op = makeDequantization(input1, dequantization1);
    auto dequantization2Op = makeDequantization(input2, dequantization2);

    std::shared_ptr<ngraph::opset1::MatMul> matMul = std::make_shared<ngraph::op::TypeRelaxed<ngraph::opset1::MatMul>>(
        std::vector<element::Type>{ element::f32, element::f32 }, std::vector<element::Type>{},
        ngraph::op::TemporaryReplaceOutputType(dequantization1Op, element::f32).get(),
        ngraph::op::TemporaryReplaceOutputType(dequantization2Op, element::f32).get(),
        false,
        false);

    matMul->set_friendly_name("matMul");
    ngraph::pass::low_precision::NetworkHelper::setOutDataPrecision(matMul, precision);
    auto dequantizationAfter = makeDequantization(matMul, resultDequantizationOperations);
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
    const FakeQuantizeOnData& fqOnData,
    const ngraph::Shape& weightsConstShape,
    const std::vector<float>& weightsConstValues,
    const FakeQuantizeOnWeights& fqOnWeights) {
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
        false);
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
