// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

DequantizationOperations::DequantizationOperations() : convertOutputPrecision(ngraph::element::undefined) {}

DequantizationOperations::DequantizationOperations(
    const ngraph::element::Type_t convertOutputPrecision,
    const std::vector<float>& subtractValues,
    const std::vector<float>& multiplyValues) :
    convertOutputPrecision(convertOutputPrecision),
    subtractValues(subtractValues),
    multiplyValues(multiplyValues)
{}

bool DequantizationOperations::empty() const {
    return (convertOutputPrecision == ngraph::element::undefined) && subtractValues.empty() && multiplyValues.empty();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
