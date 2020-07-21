// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

DequantizationOperations::Convert::Convert() :
    isEmpty(true),
    outPrecision(ngraph::element::undefined)
{}

DequantizationOperations::Convert::Convert(const ngraph::element::Type outPrecision) :
    isEmpty(false),
    outPrecision(outPrecision)
{}

bool DequantizationOperations::Convert::empty() const noexcept {
    return isEmpty;
}

DequantizationOperations::DequantizationOperations() {}

DequantizationOperations::DequantizationOperations(
    const Convert& convert,
    const std::vector<float>& subtractValues,
    const std::vector<float>& multiplyValues) :
    convert(convert),
    subtractValues(subtractValues),
    multiplyValues(multiplyValues)
{}

bool DequantizationOperations::empty() const {
    return convert.empty() && subtractValues.empty() && multiplyValues.empty();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
