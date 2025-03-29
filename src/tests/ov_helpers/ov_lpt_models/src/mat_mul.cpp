// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/mat_mul.hpp"

#include <queue>
#include <memory>

#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"
#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> MatMulFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape inputShape,
    const float low,
    const float high) {
    const auto input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantize1 = ov::test::utils::make_fake_quantize(
        input1, precision, 256ul, { 1ul },
        { low / 4.f }, { high / 4.f }, { low / 4.f }, { high / 4.f });
    fakeQuantize1->set_friendly_name("fakeQuantize1");

    const auto input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantize2 = ov::test::utils::make_fake_quantize(
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

namespace {
std::vector<float> generate_dequantization_values(
        const ov::Shape& shape,
        const size_t levels,
        const bool low) {
    const auto shape_size = ov::shape_size(shape);
    std::vector<float> values(shape_size);
    for (size_t i = 0; i < shape_size; ++i) {
        values[i] = low ? -128.f / (static_cast<float>(i) + 1.f) : 127.f / (static_cast<float>(i) + 1.f);
    }
    return values;
}
} // namespace

std::shared_ptr<ov::Model> MatMulFunction::getOriginal(
        const ov::element::Type precision,
        const ov::PartialShape& inputShape1,
        const ov::PartialShape& inputShape2,
        const bool transpose1,
        const bool transpose2,
        const bool signedOnWeights,
        const bool bias,
        const bool perChannelWeightsDequantization,
        const bool relu,
        const bool fq) {
    const auto paramNode = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    const std::vector<size_t> constShapes(inputShape1.rank().get_length(), 1ul);
    const auto fakeQuantizeOnAcitvations = signedOnWeights ?
            ov::test::utils::make_fake_quantize(
                paramNode, precision, 256ul, constShapes,
                { -128.f / 4.f }, { 127.f / 4.f }, { -128.f / 4.f }, { 127.f / 4.f }) :
            ov::test::utils::make_fake_quantize(
                paramNode, precision, 256ul, constShapes,
                { 0.f }, { 255.f / 4.f }, { 0.f }, { 255.f / 4.f });
    fakeQuantizeOnAcitvations->set_friendly_name("fakeQuantizeOnAcitvations");

    const size_t channel = inputShape2[inputShape2.size() - 2].get_length();

    // fq
    std::shared_ptr<Node> parentOnWeights;
    if (fq) {
        auto weightsConst = ov::test::utils::make_constant(precision, inputShape2.to_shape());
        parentOnWeights = perChannelWeightsDequantization ?
                          ov::test::utils::make_fake_quantize(
                                  weightsConst, precision, 256ul,
                                  Shape{channel, 1},
                                  generate_dequantization_values(Shape{channel, 1}, 256ul, true),
                                  generate_dequantization_values(Shape{channel, 1}, 256ul, false),
                                  generate_dequantization_values(Shape{channel, 1}, 256ul, true),
                                  generate_dequantization_values(Shape{channel, 1}, 256ul, false)) :
                          ov::test::utils::make_fake_quantize(
                                  weightsConst, precision, 256ul, {1ul, 1ul},
                                  {-128.f / 8.f}, {127.f / 8.f}, {-128.f / 8.f}, {127.f / 8.f});
    } else {
        Shape shape = inputShape2.to_shape();
        if (transpose2) {
            shape[shape.size() - 1ull] = 1;
        } else {
            shape[shape.size() - 2ull] = 1;
        }
        auto weightsConst = ov::test::utils::make_constant(signedOnWeights ? element::i8 : element::u8, inputShape2.to_shape(), {});
        const auto convert = std::make_shared<opset1::Convert>(weightsConst, precision);

        const auto multiplyConst = ov::test::utils::make_constant(precision, shape);
        parentOnWeights = std::make_shared<opset1::Multiply>(convert, multiplyConst);
    }

    parentOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    std::shared_ptr<Node> parent = std::make_shared<ov::opset1::MatMul>(
        fakeQuantizeOnAcitvations->output(0),
        parentOnWeights->output(0),
        transpose1,
        transpose2);
    parent->set_friendly_name("fullyConnected");

    if (bias) {
        ov::Shape bias_shape(parent->get_output_partial_shape(0).size(), 1);
        bias_shape.back() = parent->get_output_partial_shape(0).rbegin()->get_length();
        auto bias = ov::test::utils::make_constant(precision, bias_shape);
        parent = std::make_shared<ov::opset1::Add>(parent, bias);
        parent->set_friendly_name("add");
    }

    if (relu) {
        parent = std::make_shared<ov::opset1::Relu>(parent);
        parent->set_friendly_name("relu");
    }

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(parent) };
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
    const FakeQuantizeOnData& fqOnData2,
    const bool requantization) {
    const std::shared_ptr<ov::opset1::Parameter> input1 = std::make_shared<ov::opset1::Parameter>(precision, inputShape1);
    input1->set_friendly_name("input1");

    const std::shared_ptr<ov::opset1::Parameter> input2 = std::make_shared<ov::opset1::Parameter>(precision, inputShape2);
    input2->set_friendly_name("input2");

    std::shared_ptr<ov::Node> parent1 = input1;
    if (!fqOnData1.empty()) {
        parent1 = makeFakeQuantize(parent1, precision, fqOnData1);
    }

    std::shared_ptr<ov::Node> parent2 = input2;
    if (!fqOnData2.empty()) {
        parent2 = makeFakeQuantize(parent2, precision, fqOnData2);
    }

    std::shared_ptr<Node> parent = std::make_shared<ov::opset1::MatMul>(
        parent1,
        parent2,
        false,
        false);
    parent->set_friendly_name("matMul");

    if (requantization) {
        parent = makeFakeQuantize(parent, precision, fqOnData1);
        parent = std::make_shared<ov::opset1::PRelu>(
                parent,
                std::make_shared<ov::opset1::Constant>(ov::element::f32, Shape{1}, std::vector<float>{0.f}));
        parent->set_friendly_name("prelu");
    }

    std::shared_ptr<ov::opset1::Result> result = std::make_shared<ov::opset1::Result>(parent);

    std::shared_ptr<ov::Model> function = std::make_shared<ov::Model>(
        ov::ResultVector{ result },
        std::vector<std::shared_ptr<ov::op::v0::Parameter>> { input1, input2 },
        "MatMulTransformation");
    return function;
}

std::shared_ptr<ov::Model> MatMulFunction::getOriginal(const ov::element::Type netPrecision,
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
        std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
        std::vector<ov::element::Type>{ov::element::f32},
        ov::op::TemporaryReplaceOutputType(dequantization1Op, ov::element::f32).get(),
        ov::op::TemporaryReplaceOutputType(dequantization2Op, ov::element::f32).get(),
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
        std::vector<ov::element::Type>{ov::element::f32, ov::element::f32},
        std::vector<ov::element::Type>{},
        ov::op::TemporaryReplaceOutputType(lastDequantizationBefore, ov::element::f32).get(),
        ov::op::TemporaryReplaceOutputType(weightsConst, ov::element::f32).get(),
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
}  // namespace ov
