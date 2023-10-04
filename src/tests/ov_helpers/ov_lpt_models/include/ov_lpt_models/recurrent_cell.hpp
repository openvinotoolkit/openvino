// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <algorithm>
#include <memory>
#include <ngraph/ngraph.hpp>
#include "low_precision/layer_transformation.hpp"
#include "common/fake_quantize_on_data.hpp"
#include "common/dequantization_operations.hpp"

namespace ngraph {
namespace builder {
namespace subgraph {

class RecurrentCellFunction {
public:
    enum class RNNType { LSTMSequence, GRUSequence };

    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type inputPrecision,
        const std::vector<ngraph::PartialShape>& inputActivationsShapes,
        const std::vector<ngraph::Shape>& inputWeightsShapes,
        const RNNType type,
        const std::vector<FakeQuantizeOnDataWithConstant>& fqOnDatas,
        const std::vector<DequantizationOperations::Convert>& converts,
        const std::vector<DequantizationOperations>& dequantizations);
};

std::shared_ptr<Node> makeQuantizationAndDequantization(const std::shared_ptr<Node> input,
                                                        const ngraph::element::Type inputPrecision,
                                                        const std::string friendly_name,
                                                        const FakeQuantizeOnDataWithConstant& fqOnData,
                                                        const DequantizationOperations::Convert& convert,
                                                        const DequantizationOperations& dequantization);
}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
