// Copyright (C) 2018-2023 Intel Corporation
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
        const std::vector<ngraph::PartialShape>& inputShape,
        const size_t concatInputsCount,
        const std::vector<FakeQuantizeOnDataWithConstant>& fqBefore,
        const DequantizationOperations::Convert& convertBefore,
        const DequantizationOperations& dequantizationBefore,
        const std::string& operation,
        const FakeQuantizeOnDataWithConstant& fqOnDataAfter,
        const DequantizationOperations::Convert& convertAfter,
        const DequantizationOperations& dequantizationAfter,
        const std::vector<ov::Any>& concatAttributes,
        const ngraph::element::Type precisionAfterOperation,
        const std::int64_t& axis,
        const bool oneInputWithSplit);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
