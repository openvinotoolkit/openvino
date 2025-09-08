// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/op/constant.hpp"
#include "ov_lpt_models/common/fake_quantize_on_weights.hpp"
#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "ov_lpt_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class ConvolutionBackpropDataFunction {
public:
    static std::shared_ptr<Node> getWeights(
        const Shape& shape,
        const ov::element::Type& netPrecision,
        const builder::subgraph::DequantizationOperations& dequantizationOnWeights,
        const std::shared_ptr<ov::op::v0::Constant>& value = nullptr);
    static std::shared_ptr<Node> getWeights(
        const Shape& shape,
        const ov::element::Type& netPrecision,
        const builder::subgraph::FakeQuantizeOnWeights& fqOnWeights,
        const std::shared_ptr<ov::op::v0::Constant>& value = nullptr);
    static std::shared_ptr<Node> getWeights(
        const Shape& shape,
        const ov::element::Type& netPrecision,
        const builder::subgraph::FakeQuantizeOnWeights& fqOnWeights,
        const builder::subgraph::DequantizationOperations& dequantizationOnWeights,
        const std::shared_ptr<ov::op::v0::Constant>& value = nullptr);
    static std::shared_ptr<ov::Model> get(
        const ov::element::Type netPrecision,
        const PartialShape& inputShape,
        const Shape& outputShape,
        const builder::subgraph::FakeQuantizeOnData& fqOnData,
        const std::shared_ptr<Node>& weights);
    static std::shared_ptr<ov::Model> getOriginal(
        const ov::element::Type precision,
        const ov::element::Type netPrecision,
        const PartialShape& inputShape,
        const Shape& outputShape,
        const builder::subgraph::DequantizationOperations& dequantization,
        const std::shared_ptr<Node>& weights);
    static std::shared_ptr<ov::Model> getReference(
        const ov::element::Type precision,
        const ov::element::Type netPrecision,
        const PartialShape& inputShape,
        const Shape& outputShape,
        const builder::subgraph::DequantizationOperations& dequantization,
        const std::shared_ptr<Node>& weights,
        const builder::subgraph::DequantizationOperations& dequantizationAfter);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
