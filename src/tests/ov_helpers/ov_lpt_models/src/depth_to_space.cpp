// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/depth_to_space.hpp"

#include "ov_lpt_models/common/builders.hpp"
#include "common_test_utils/node_builders/fake_quantize.hpp"

namespace ov {
namespace builder {
namespace subgraph {

std::shared_ptr<ov::Model> DepthToSpaceFunction::getOriginal(
    const ov::element::Type precision,
    const ov::PartialShape& inputShape,
    const ov::opset1::DepthToSpace::DepthToSpaceMode mode,
    const size_t blockSize) {
    const float low = 0.f;
    const float high = 255.f;
    const float inputScale = 10.f;
    const float outputScale = 20.f;

    const auto input = std::make_shared<ov::opset1::Parameter>(precision, inputShape);

    const auto fakeQuantize = ov::test::utils::make_fake_quantize(
        input, precision, 256, { 1, 1, 1, 1 },
        { low / inputScale }, { high / inputScale }, { low / outputScale }, { high / outputScale });

    auto d2s = std::make_shared<ov::opset1::DepthToSpace>(fakeQuantize, mode, blockSize);
    d2s->set_friendly_name("output");

    ov::ResultVector results = { std::make_shared<ov::opset1::Result>(d2s) };

    const auto function = std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "DepthToSpaceTransformation");
    return function;
}

std::shared_ptr<ov::Model> DepthToSpaceFunction::getOriginal(
    const ov::PartialShape& inputShape,
    const ov::opset1::DepthToSpace::DepthToSpaceMode mode,
    const size_t blockSize,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantization) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const auto dequantizationOp = makeDequantization(input, dequantization);
    auto d2s = std::make_shared<ov::opset1::DepthToSpace>(dequantizationOp, mode, blockSize);
    d2s->set_friendly_name("output");

    ov::ResultVector results = { std::make_shared<ov::opset1::Result>(d2s) };

    const auto function = std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "DepthToSpaceTransformation");
    return function;
}

std::shared_ptr<ov::Model> DepthToSpaceFunction::getReference(
    const ov::PartialShape& inputShape,
    const ov::opset1::DepthToSpace::DepthToSpaceMode mode,
    const size_t blockSize,
    const ov::element::Type precisionBeforeDequantization,
    const ov::builder::subgraph::DequantizationOperations& dequantizationBefore,
    const ov::element::Type precisionAfterOperation,
    const ov::builder::subgraph::DequantizationOperations& dequantizationAfter) {
    const auto input = std::make_shared<ov::opset1::Parameter>(precisionBeforeDequantization, inputShape);

    const std::shared_ptr<Node> dequantizationOpBefore = makeDequantization(input, dequantizationBefore);
    auto d2s = std::make_shared<ov::opset1::DepthToSpace>(dequantizationOpBefore, mode, blockSize);
    const std::shared_ptr<Node> dequantizationOpAfter = makeDequantization(d2s, dequantizationAfter);
    dequantizationOpAfter->set_friendly_name("output");

    ov::ResultVector results = { std::make_shared<ov::opset1::Result>(dequantizationOpAfter) };
    ov::pass::low_precision::NetworkHelper::setOutDataPrecision(d2s, precisionAfterOperation);

    const auto function = std::make_shared<ov::Model>(results, ov::ParameterVector{ input }, "DepthToSpaceTransformation");
    return function;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
