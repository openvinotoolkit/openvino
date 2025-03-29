// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/dequantization_operations.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class PrecisionPropagationFunction {
public:
    static std::shared_ptr<ov::Model> getOriginalWithNeighbors(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const DequantizationOperations::Convert& convert1,
        const DequantizationOperations& dequantization1,
        const FakeQuantizeOnData& fqOnData2,
        const DequantizationOperations::Convert& convert2,
        const DequantizationOperations& dequantization2,
        const FakeQuantizeOnData& fqOnData3,
        const DequantizationOperations::Convert& convert3,
        const DequantizationOperations& dequantization3);

    static std::shared_ptr<ov::Model> getReferenceWithNeighbors(
        const ov::element::Type precision,
        const ov::Shape& inputShape,
        const FakeQuantizeOnData& fqOnData1,
        const FakeQuantizeOnData& fqOnData2,
        const FakeQuantizeOnData& fqOnData3,
        const ov::element::Type precisionBeforeOp,
        const DequantizationOperations& dequantizationBefore,
        const ov::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationOperations1,
        const DequantizationOperations& dequantizationOperations2);

private:
    static std::shared_ptr<Node> makeMaxPool(const ov::Output<Node>& parent, const std::vector<size_t>& kernel);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
