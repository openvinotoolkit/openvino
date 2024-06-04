// Copyright (C) 2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/fuse_fake_quantize.hpp"

#include "openvino/opsets/opset1.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "low_precision/network_helper.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

using namespace ov::pass;

std::shared_ptr<ov::Model> FuseFakeQuantizeFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBeforeAdd,
    const Add& add,
    const ov::element::Type precisionBeforeDequantization,
    const DequantizationOperations& dequantization,
    const ov::element::Type precisionAfterDequantization,
    const ov::element::Type precisionFqOnData,
    const FakeQuantizeOnDataWithConstant& fqOnData) {
    const auto input = std::make_shared<ov::opset1::Parameter>(add.empty() ? precisionBeforeDequantization : precisionBeforeAdd, inputShape);
    input->set_friendly_name("input");

    std::shared_ptr<Node> parent = input;
    if (!add.empty()) {
        parent = makeElementwise<ov::opset1::Add>(parent, add);
    }

    const std::shared_ptr<Node> lastDequantization = makeDequantization(parent, dequantization);

    const std::shared_ptr<Node> fakeQuantize = precisionAfterDequantization == precisionFqOnData ?
        makeFakeQuantize(lastDequantization, precisionFqOnData, fqOnData) :
        makeFakeQuantizeTypeRelaxed(lastDequantization, precisionFqOnData, fqOnData);
    fakeQuantize->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(fakeQuantize) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FuseFakeQuantizeFunction");
}

namespace {
std::shared_ptr<ov::opset1::Convolution> make_convolution(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionData,
    const ov::element::Type precisionWeights,
    const std::shared_ptr<Node>& parent,
    const size_t index) {
    const ov::Shape shape = inputShape.to_shape();
    const ov::Shape weightsShape({ shape[1], shape[1], 1ull, 1ull });
    auto weightsConstant = std::make_shared<ov::op::v0::Constant>(precisionWeights, weightsShape, std::vector<float>(9, 1.f));
    auto weights = makeFakeQuantize(
        weightsConstant,
        precisionData,
        FakeQuantizeOnData(
            255,
            ov::Shape({ shape[1], 1ull, 1ull, 1ull }),
            { -1.27f, -1.27f, -1.27f },
            { 1.28f, 1.28f, 1.28f },
            { -1.27f, -1.27f, -1.27f },
            { 1.28f, 1.28f, 1.28f },
            precisionData));

    auto convolution = std::make_shared<ov::opset1::Convolution>(
        parent,
        weights,
        ov::Strides{ 1, 1 },
        ov::CoordinateDiff{ 0, 0 },
        ov::CoordinateDiff{ 0, 0 },
        ov::Strides{ 1, 1 });
    convolution->set_friendly_name("convolution" + std::to_string(index));
    return convolution;
}
}  // namespace

    std::shared_ptr<ov::Model> FuseFakeQuantizeFunction::getReference(
        const ov::PartialShape& inputShape,
        const ov::element::Type precisionBeforeAdd,
        const Add& add,
        const ov::element::Type precisionBeforeDequantization,
        const DequantizationOperations& dequantization,
        const ov::element::Type precisionAfterDequantization,
        const ov::element::Type precisionFqOnData,
        const FakeQuantizeOnDataWithConstant& fqOnData) {
        const auto input = std::make_shared<ov::opset1::Parameter>(add.empty() ? precisionBeforeDequantization : precisionBeforeAdd, inputShape);
        input->set_friendly_name("input");

        std::shared_ptr<Node> parent = input;
        if (!add.empty()) {
            parent = makeElementwise<ov::opset1::Add>(parent, add);
        }

        const std::shared_ptr<Node> lastDequantization = makeDequantization(parent, dequantization);

        auto fqOnDataCopy = fqOnData;
        fqOnDataCopy.outputHighValues = {255.f};
        fqOnDataCopy.outputPrecision =
            fqOnData.outputPrecision == ov::element::undefined ? ov::element::u8 : fqOnData.outputPrecision;

        std::shared_ptr<Node> lastNode = makeFakeQuantizeTypeRelaxed(lastDequantization, precisionFqOnData, fqOnDataCopy);
        lastNode = makeDequantization(lastNode,
                                      {lastNode->output(0).get_element_type() != ov::element::f32
                                           ? DequantizationOperations::Convert{ov::element::f32}
                                           : DequantizationOperations::Convert{},
                                       {},
                                       {{0.01f}, precisionFqOnData}});
        lastNode->set_friendly_name("output");

        ov::ResultVector results{ std::make_shared<ov::opset1::Result>(lastNode) };
        return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FuseFakeQuantizeFunction");
    }

std::shared_ptr<ov::Model> FuseFakeQuantizeFunction::get(
    const ov::PartialShape& inputShape,
    const ov::element::Type precisionBefore,
    const FakeQuantizeOnData& fqOnData1,
    const FakeQuantizeOnData& fqOnData2,
    const DequantizationOperations& dequantizationOperations2) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBefore, inputShape);
    input->set_friendly_name("input");

    std::shared_ptr<Node> parent = input;

    if (!fqOnData1.empty()) {
        parent = fqOnData1.outputPrecision == precisionBefore ?
            makeFakeQuantize(parent, precisionBefore, fqOnData1) :
            makeFakeQuantizeTypeRelaxed(parent, precisionBefore, fqOnData1);
        parent->set_friendly_name("fakeQuantize1");
    }

    const std::vector<size_t> kernel = { 3, 3 };
    const std::vector<size_t> stride = { 1, 1 };
    const std::vector<size_t> padBegin = { 0, 0 };
    const std::vector<size_t> padEnd = { 0, 0 };
    const ov::op::PadType padType = ov::op::PadType::NOTSET;
    const ov::op::RoundingType roundingType = ov::op::RoundingType::FLOOR;

    parent = std::make_shared<ov::opset1::MaxPool>(
        parent,
        stride,
        padBegin,
        padEnd,
        kernel,
        roundingType,
        padType);

    if (!fqOnData2.empty()) {
        parent = makeFakeQuantize(parent, precisionBefore, fqOnData2);
        parent->set_friendly_name("fakeQuantize2");
    }

    if (!dequantizationOperations2.empty()) {
        parent = makeDequantization(parent, dequantizationOperations2);
    }

    ov::ResultVector results{
        std::make_shared<ov::opset1::Result>(make_convolution(inputShape, precisionBefore, precisionBefore, parent, 0)),
        std::make_shared<ov::opset1::Result>(make_convolution(inputShape, precisionBefore, precisionBefore, parent, 1))
    };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "FuseFakeQuantizeFunction");
}

std::shared_ptr<ov::Model> FuseFakeQuantizeFunction::get(
    const ov::PartialShape& inputShape,
    const std::vector<Branch>& branches,
    const ov::element::Type precisionFqOnData,
    const FakeQuantizeOnData& fqOnData) {
    if (branches.size() != 2ul) {
        throw std::runtime_error("unsupported branches count");
    }

    if (branches[0].dequantization.multiply.outPrecision != branches[1].dequantization.multiply.outPrecision) {
        throw std::runtime_error("branch precisions are not equal");
    }

    ov::ParameterVector inputs;
    std::vector<std::shared_ptr<Node>> lastDequantizations;
    for (const Branch& branch : branches) {
        const auto input = std::make_shared<ov::opset1::Parameter>(branch.precisionBeforeDequantization, inputShape);
        inputs.push_back(input);

        const std::shared_ptr<Node> lastDequantization = makeDequantization(input, branch.dequantization);
        lastDequantizations.push_back(lastDequantization);
    }

    std::shared_ptr<ov::opset1::Multiply> multiply = std::make_shared<ov::opset1::Multiply>(lastDequantizations[0], lastDequantizations[1]);

    const std::shared_ptr<Node> fakeQuantize = branches[0].dequantization.multiply.outPrecision == precisionFqOnData ?
        makeFakeQuantize(multiply, precisionFqOnData, fqOnData) :
        makeFakeQuantizeTypeRelaxed(multiply, precisionFqOnData, fqOnData);
    fakeQuantize->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(fakeQuantize) };
    return std::make_shared<ov::Model>(results, inputs, "FuseFakeQuantizeFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
