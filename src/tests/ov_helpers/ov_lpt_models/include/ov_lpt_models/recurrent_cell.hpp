// Copyright (C) 2022-2024 Intel Corporation
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

class RecurrentCellFunction {
public:
    enum class RNNType { LSTMSequence, GRUSequence };

    static std::shared_ptr<ov::Model> get(
        const ov::element::Type inputPrecision,
        const std::vector<ov::PartialShape>& inputActivationsShapes,
        const std::vector<ov::Shape>& inputWeightsShapes,
        const RNNType type,
        const std::vector<FakeQuantizeOnDataWithConstant>& fqOnDatas,
        const std::vector<DequantizationOperations::Convert>& converts,
        const std::vector<DequantizationOperations>& dequantizations,
        const bool addPrecisionTransparentOperations = false);
};

std::shared_ptr<Node> makeQuantizationAndDequantization(const std::shared_ptr<Node> input,
                                                        const ov::element::Type inputPrecision,
                                                        const std::string friendly_name,
                                                        const FakeQuantizeOnDataWithConstant& fqOnData,
                                                        const DequantizationOperations::Convert& convert,
                                                        const DequantizationOperations& dequantization,
                                                        const bool addPrecisionTransparentOperations = false);
}  // namespace subgraph
}  // namespace builder
}  // namespace ov
