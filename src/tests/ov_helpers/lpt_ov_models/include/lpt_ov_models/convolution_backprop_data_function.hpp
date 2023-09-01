// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once


#include <openvino/opsets/opset1.hpp>

#include "lpt_ov_models/common/fake_quantize_on_weights.hpp"
#include "lpt_ov_models/common/fake_quantize_on_data.hpp"
#include "lpt_ov_models/common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class ConvolutionBackpropDataFunction {
public:
    static std::shared_ptr<Node> getWeights(
        const Shape& shape,
        const element::Type& netPrecision,
        const builder::subgraph::DequantizationOperations& dequantizationOnWeights,
        const std::shared_ptr<opset1::Constant>& value = nullptr);
    static std::shared_ptr<Node> getWeights(
        const Shape& shape,
        const element::Type& netPrecision,
        const builder::subgraph::FakeQuantizeOnWeights& fqOnWeights,
        const std::shared_ptr<opset1::Constant>& value = nullptr);
    static std::shared_ptr<Node> getWeights(
        const Shape& shape,
        const element::Type& netPrecision,
        const builder::subgraph::FakeQuantizeOnWeights& fqOnWeights,
        const builder::subgraph::DequantizationOperations& dequantizationOnWeights,
        const std::shared_ptr<opset1::Constant>& value = nullptr);
    static std::shared_ptr<Model> get(
        const element::Type netPrecision,
        const PartialShape& inputShape,
        const Shape& outputShape,
        const builder::subgraph::FakeQuantizeOnData& fqOnData,
        const std::shared_ptr<Node>& weights);
    static std::shared_ptr<Model> getOriginal(
        const element::Type precision,
        const element::Type netPrecision,
        const PartialShape& inputShape,
        const Shape& outputShape,
        const builder::subgraph::DequantizationOperations& dequantization,
        const std::shared_ptr<Node>& weights);
    static std::shared_ptr<Model> getReference(
        const element::Type precision,
        const element::Type netPrecision,
        const PartialShape& inputShape,
        const Shape& outputShape,
        const builder::subgraph::DequantizationOperations& dequantization,
        const std::shared_ptr<Node>& weights,
        const builder::subgraph::DequantizationOperations& dequantizationAfter);
};
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
