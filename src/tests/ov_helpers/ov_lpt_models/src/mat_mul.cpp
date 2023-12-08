// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/mat_mul.hpp"

#include <queue>
#include <memory>

#include <openvino/opsets/opset1.hpp>
#include "ov_ops/type_relaxed.hpp"
#include "ov_models/subgraph_builders.hpp"
#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> MatMulFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape inputShape,
    const float low,
    const float high) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantize1 = ngraph::builder::makeFakeQuantize(
        input1, precision, 256ul, { 1ul },
        { low / 4.f }, { high / 4.f }, { low / 4.f }, { high / 4.f });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantize2 = ngraph::builder::makeFakeQuantize(
        input2, precision, 256ul, { 1ul },
        { low / 8.f }, { high / 8.f }, { low / 8.f }, { high / 8.f });
    fakeQuantize2->set_friendly_name("fakeQuantize2");

    const auto matMul = std::make_shared<ov::opset1::MatMul>(
        fakeQuantize1->output(0),
        fakeQuantize2->output(0),
        false,
        false);
    matMul->set_friendly_name("matMul");

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ std::make_shared<ov::opset1::Result>(matMul) },
        ov::ParameterVector{ input1, input2 },
        "GemmTransformation");

    return function;
}

std::shared_ptr<ov::Model> MatMulFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape inputShape1,
    const ov::PartialShape inputShape2,
    const bool transpose1,
    const bool transpose2) {
    const auto paramNode = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    const std::vector<size_t> constShapes(inputShape1.rank().get_length(), 1ul);
    const auto fakeQuantizeOnAcitvations = ngraph::builder::makeFakeQuantize(
        paramNode, precision, 256ul, constShapes,
        { 0.f }, { 255.f / 4.f }, { 0.f }, { 255.f / 4.f });
    fakeQuantizeOnAcitvations->set_friendly_name("fakeQuantizeOnAcitvations");

    auto weightsConst = std::make_shared<ov::op::v0::Constant>(
        precision,
        inputShape2.to_shape(),
        std::vector<float>({ 1.f }));
    const auto fakeQuantizeOnWeights = ngraph::builder::makeFakeQuantize(
        weightsConst, precision, 256ul, { 1ul, 1ul },
        { -128.f / 8.f }, { 127.f / 8.f }, { -128.f / 8.f }, { 127.f / 8.f });
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    const std::shared_ptr<ov::opset1::MatMul> fullyConnected = std::make_shared<ov::opset1::MatMul>(
        fakeQuantizeOnAcitvations->output(0),
        fakeQuantizeOnWeights->output(0),
        transpose1,
        transpose2);
    fullyConnected->set_friendly_name("fullyConnected");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(fullyConnected) };
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        results,
        ov::ParameterVector{ paramNode },
        "FullyConnectedTransformation");

    return function;
}


std::shared_ptr<ov::Model> MatMulFunction::getOriginal(
    const ov::element::Type precision,
    const ov::Shape& inputShape1,
    const FakeQuantizeOnData& fqOnData1,
    const ov::Shape& inputShape2,
    const FakeQuantizeOnData& fqOnData2) {
    const std::shared_ptr<ov::opset1::Parameter> input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    input1->set_friendly_name("input1");

    const std::shared_ptr<ov::opset1::Parameter> input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape2);
    input2->set_friendly_name("input2");

    const std::shared_ptr<ov::opset1::MatMul> matMul = std::make_shared<ov::opset1::MatMul>(
        makeFakeQuantize(input1, precision, fqOnData1),
        makeFakeQuantize(input2, precision, fqOnData2),
        false,
        false);
    matMul->set_friendly_name("matMul");

    std::shared_ptr<ov::opset1::Result> result = std::make_shared<ov::opset1::Result>(matMul);

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input1, input2 },
        "MatMulTransformation");
    return function;
}

std::shared_ptr<ov::Model> MatMulFunction::getOriginal(
    const element::Type netPrecision,
    const ov::PartialShape& inputShape1,
    const ov::element::Type precisionBeforeDequantization1,
    const DequantizationOperations& dequantization1,
    const ov::PartialShape& inputShape2,
    const ov::element::Type precisionBeforeDequantization2,
    const DequantizationOperations& dequantization2) {
    if (!dequantization1.convert.empty() && (precisionBeforeDequantization1 == dequantization1.convert.outPrecision)) {
        throw std::runtime_error("unexpected input arguments for branch 1");
    }

    if (!dequantization2.convert.empty() && (precisionBeforeDequantization2 == dequantization2.convert.outPrecision)) {
        throw std::runtime_error("unexpected input arguments for branch 2");
    }

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization1, inputShape1);
    input1->set_friendly_name("input1");

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization2, inputShape2);
    input2->set_friendly_name("input2");

    DequantizationOperations deqSructure1 = dequantization1;
    deqSructure1.multiply.outPrecision = netPrecision;
    DequantizationOperations deqSructure2 = dequantization2;
    deqSructure2.multiply.outPrecision = netPrecision;

    const std::shared_ptr<ov::opset1::MatMul> matMul = std::make_shared<ov::opset1::MatMul>(
        makeDequantization(input1, deqSructure1),
        makeDequantization(input2, deqSructure2),
        false,
        false);
    matMul->set_friendly_name("matMul");

    std::shared_ptr<ov::opset1::Result> result = std::make_shared<ov::opset1::Result>(matMul);

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input1, input2 },
        "MatMulTransformation");
    return function;
}

std::shared_ptr<ov::Model> getOriginalWithConstant2(
    const ov::element::Type precision) {
    return nullptr;
}

std::shared_ptr<ov::Model> MatMulFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const DequantizationOperations& deqOnData,
    const Constant& weights,
    const FakeQuantizeOnWeights& fqOnWeights,
    const DequantizationOperations& deqOnWeights) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    input->set_friendly_name("input1");

    const auto dequantizationOnData = makeDequantization(input, deqOnData);

    const std::shared_ptr<ov::Node> weightsConst = std::make_shared<ov::opset1::Constant>(
        weights.outPrecision,
        weights.shape,
        weights.values);

    const std::shared_ptr<ov::Node> fakeQuantize = fqOnWeights.empty() ? nullptr : makeFakeQuantize(weightsConst, precision, fqOnWeights);
    const auto dequantizationOnWeights = makeDequantization(fakeQuantize == nullptr ? weightsConst : fakeQuantize, deqOnWeights);

    const auto matMul = std::make_shared<ov::opset1::MatMul>(
        dequantizationOnData,
        dequantizationOnWeights,
        false,
        false);
    matMul->set_friendly_name("matMul");
    auto& rtInfo = matMul->get_rt_info();
    rtInfo["Variant::std::string"] = "matMul";

    const auto result = std::make_shared<ov::opset1::Result>(matMul);
    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input },
        "MatMulTransformation");

    return function;
}

std::shared_ptr<ov::Model> MatMulFunction::getReference(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape1,
    const ov::element::Type precisionBeforeDequantization1,
    const DequantizationOperations& dequantization1,
    const ov::PartialShape& inputShape2,
    const ov::element::Type precisionBeforeDequantization2,
    const DequantizationOperations& dequantization2,
    const DequantizationOperations& resultDequantizationOperations) {
    if (!dequantization1.convert.empty() && (precisionBeforeDequantization1 == dequantization1.convert.outPrecision)) {
        throw std::runtime_error("unexpected input arguments for branch 1");
    }

    if (!dequantization2.convert.empty() && (precisionBeforeDequantization2 == dequantization2.convert.outPrecision)) {
        throw std::runtime_error("unexpected input arguments for branch 2");
    }

    const auto input1 = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization1, inputShape1);
    input1->set_friendly_name("input1");

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization2, inputShape2);
    input2->set_friendly_name("input2");

    DequantizationOperations deqSructure1 = dequantization1;
    deqSructure1.multiply.outPrecision = precision;
    DequantizationOperations deqSructure2 = dequantization2;
    deqSructure2.multiply.outPrecision = precision;

    auto dequantization1Op = makeDequantization(input1, deqSructure1);
    auto dequantization2Op = makeDequantization(input2, deqSructure2);

    std::shared_ptr<ov::opset1::MatMul> matMul = std::make_shared<ov::op::TypeRelaxed<ov::opset1::MatMul>>(
        std::vector<element::Type>{ element::f32, element::f32 }, std::vector<element::Type>{ element::f32 },
        ov::op::TemporaryReplaceOutputType(dequantization1Op, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(dequantization2Op, element::f32).get(),
        false,
        false);

    matMul->set_friendly_name("matMul");
    DequantizationOperations deqSructureAfter = resultDequantizationOperations;
    deqSructureAfter.multiply.outPrecision = precision;
    auto dequantizationAfter = makeDequantization(matMul, deqSructureAfter);
    dequantizationAfter->set_friendly_name("matMul");

    std::shared_ptr<ov::opset1::Result> result = std::make_shared<ov::opset1::Result>(dequantizationAfter);

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input1, input2 },
        "MatMulTransformation");
    return function;
}

std::shared_ptr<ov::Model> MatMulFunction::getReference(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeDequantization,
    const DequantizationOperations& dequantization,
    const Constant& weights,
    const DequantizationOperations& resultDequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);
    input->set_friendly_name("input1");

    const std::shared_ptr<ov::Node> lastDequantizationBefore = makeDequantization(input, dequantization);

    const std::shared_ptr<ov::opset1::Constant> weightsConst = std::make_shared<ov::opset1::Constant>(
        weights.outPrecision,
        weights.shape,
        weights.values);

    const std::shared_ptr<ov::opset1::MatMul> matMul = std::make_shared<ov::op::TypeRelaxed<ov::opset1::MatMul>>(
        std::vector<ov::element::Type>{ element::f32, element::f32 }, std::vector<ov::element::Type>{},
        ov::op::TemporaryReplaceOutputType(lastDequantizationBefore, element::f32).get(),
        ov::op::TemporaryReplaceOutputType(weightsConst, element::f32).get(),
        false,
        false);
    matMul->set_friendly_name("matMul");
    auto& rtInfo = matMul->get_rt_info();
    rtInfo["Variant::std::string"] = "matMul";
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(matMul, precision);

    const std::shared_ptr<ov::Node> lastDequantizationAfter = makeDequantization(matMul, resultDequantization);
    lastDequantizationAfter->set_friendly_name("matMul");

    std::shared_ptr<ov::opset1::Result> result = std::make_shared<ov::opset1::Result>(lastDequantizationAfter);

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input },
        "MatMulTransformation");
    return function;
}

std::shared_ptr<ov::Model> MatMulFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const FakeQuantizeOnDataWithConstant& fqOnData,
    const Constant& weights,
    const FakeQuantizeOnDataWithConstant& fqOnWeights,
    const DequantizationOperations& deqOnWeights) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    input->set_friendly_name("input1");

    const auto dequantizationOnData = makeFakeQuantize(input, precision, fqOnData);

    const std::shared_ptr<ov::Node> weightsConst = std::make_shared<ov::opset1::Constant>(
        weights.outPrecision.is_real() ? precision : weights.outPrecision,
        weights.shape,
        weights.values);

    const std::shared_ptr<ov::Node> fakeQuantize = fqOnWeights.empty() ? nullptr : makeFakeQuantize(weightsConst, precision, fqOnWeights);

    auto deqStructure = deqOnWeights;
    deqStructure.setPrecision(precision);
    const auto dequantizationOnWeights = makeDequantization(fakeQuantize == nullptr ? weightsConst : fakeQuantize, deqStructure);

    const std::shared_ptr<ov::opset1::MatMul> matMul = std::make_shared<ov::opset1::MatMul>(
        dequantizationOnData,
        dequantizationOnWeights,
        false,
        true);
    matMul->set_friendly_name("matMul");

    const std::shared_ptr<ov::opset1::Result> result = std::make_shared<ov::opset1::Result>(matMul);
    result->set_friendly_name("result");

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input },
        "MatMulTransformation");

    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
