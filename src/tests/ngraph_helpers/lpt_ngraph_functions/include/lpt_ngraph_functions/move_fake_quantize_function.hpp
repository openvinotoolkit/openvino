// Copyright (C) 2021 Intel Corporation
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

class MoveFakeQuantize {
public:
    static std::shared_ptr<ngraph::Function> get(
        const ngraph::element::Type inputPrecision,
        const ngraph::PartialShape& inputShape,
        const FakeQuantizeOnDataWithConstant& fqOnData1,
        const DequantizationOperations::Convert& convert1,
        const DequantizationOperations& dequantization1,
        const FakeQuantizeOnDataWithConstant& fqOnData2,
        const DequantizationOperations::Convert& convert2,
        const DequantizationOperations& dequantization2,
        const std::string& operation,
        const FakeQuantizeOnDataWithConstant& fqOnData3,
        const DequantizationOperations::Convert& convert3,
        const DequantizationOperations& dequantization3,
        const std::vector<ov::Any>& concatAttributes,
        const ngraph::element::Type precisionAfterOperation,
        const DequantizationOperations& dequantizationAfter,
        const std::int64_t& axis);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
