// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/interpolate.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> InterpolateFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::Shape& outputShape,
    const ov::op::v0::Interpolate::Attributes& interpAttrs,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const auto dequantizationOp = makeDequantization(input, dequantization);
    const auto outShape = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ outputShape.size() }, outputShape);
    const auto interpolate = std::make_shared<ov::opset1::Interpolate>(dequantizationOp, outShape, interpAttrs);
    interpolate->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(interpolate) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "InterpolateFunction");
}

std::shared_ptr<ov::Model> InterpolateFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ov::Shape& outputShape,
    const ov::op::v0::Interpolate::Attributes& interpAttrs) {
    float k = 50.f;

    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantizeOnActivations = ov::test::utils::make_fake_quantize(
        input, precision, 256ul, { 1ul },
        { 0.f }, { 255.f / k }, { 10.f }, { 255.f / k });
    const auto outShape = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ outputShape.size() }, outputShape);
    const auto interpolate = std::make_shared<ov::opset1::Interpolate>(fakeQuantizeOnActivations, outShape, interpAttrs);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(interpolate) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "InterpolateFunction");
}

std::shared_ptr<ov::Model> InterpolateFunction::getReference(
    const ov::PartialShape& inputShape,
    const ov::Shape& outputShape,
    const ov::op::v0::Interpolate::Attributes& interpAttrs,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);
    const auto outShape = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ outputShape.size() }, outputShape);
    const auto interpolate = std::make_shared<ov::opset1::Interpolate>(quantizationOpBefore, outShape, interpAttrs);
    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(interpolate, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "InterpolateFunction");
}

// v4:interpolate
std::shared_ptr<ov::Model> InterpolateFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::Shape& outputShape,
    const ov::Shape& scalesShape,
    const ov::op::v4::Interpolate::InterpolateAttrs& interpAttrs,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const auto dequantizationOp = makeDequantization(input, dequantization);
    const auto outShape = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ outputShape.size() }, outputShape);
    const auto scales = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{ scalesShape.size() }, scalesShape);
    const auto interpolate = std::make_shared<ov::op::v4::Interpolate>(dequantizationOp, outShape, scales, interpAttrs);
    interpolate->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(interpolate) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "InterpolateFunction");
}

std::shared_ptr<ov::Model> InterpolateFunction::getOriginal(
    const ov::element::Type precision,
    const ov::Shape& inputShape,
    const ov::Shape& outputShape,
    const ov::Shape& scalesShape,
    const ov::op::v4::Interpolate::InterpolateAttrs& interpAttrs) {
    float k = 50.f;

    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);
    const auto fakeQuantizeOnActivations = ov::test::utils::make_fake_quantize(
        input, precision, 256ul, { 1ul },
        { 0.f }, { 255.f / k }, { 10.f }, { 255.f / k });
    const auto outShape = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ outputShape.size() }, outputShape);
    const auto scales = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{ scalesShape.size() }, scalesShape);
    const auto interpolate = std::make_shared<ov::op::v4::Interpolate>(fakeQuantizeOnActivations, outShape, scales, interpAttrs);

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(interpolate) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "InterpolateFunction");
}

std::shared_ptr<ov::Model> InterpolateFunction::getReference(
    const ov::PartialShape& inputShape,
    const ov::Shape& outputShape,
    const ov::Shape& scalesShape,
    const ov::op::v4::Interpolate::InterpolateAttrs& interpAttrs,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> quantizationOpBefore = makeDequantization(input, dequantizationBefore);
    const auto outShape = std::make_shared<ov::opset1::Constant>(ov::element::i64, ov::Shape{ outputShape.size() }, outputShape);
    const auto scales = std::make_shared<ov::opset1::Constant>(ov::element::f32, ov::Shape{ scalesShape.size() }, scalesShape);
    const auto interpolate = std::make_shared<ov::op::v4::Interpolate>(quantizationOpBefore, outShape, scales, interpAttrs);
    const std::shared_ptr<Node> quantizationOpAfter = makeDequantization(interpolate, dequantizationAfter);
    quantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results{ std::make_shared<ov::opset1::Result>(quantizationOpAfter) };
    return std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "InterpolateFunction");
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
