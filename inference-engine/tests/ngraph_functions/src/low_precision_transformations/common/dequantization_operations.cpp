// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ngraph_functions/low_precision_transformations/common/dequantization_operations.hpp"
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

DequantizationOperations::Convert::Convert() :
    _isEmpty(true),
    outPrecision(ngraph::element::undefined)
{}

DequantizationOperations::Convert::Convert(const ngraph::element::Type outPrecision) :
    _isEmpty(false),
    outPrecision(outPrecision)
{}

bool DequantizationOperations::Convert::empty() const noexcept {
    return _isEmpty;
}

DequantizationOperations::Subtract::Subtract() :
    _isEmpty(true),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false)
{}

DequantizationOperations::Subtract::Subtract(const float value, const bool addDequantizationAttribute) :
    _isEmpty(false),
    values({ value }),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false),
    addDequantizationAttribute(addDequantizationAttribute) {
}

DequantizationOperations::Subtract::Subtract(const std::vector<float>& values, const bool addDequantizationAttribute) :
    _isEmpty(values.empty()),
    values(values),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false),
    addDequantizationAttribute(addDequantizationAttribute) {
}

DequantizationOperations::Subtract::Subtract(const std::vector<float>& values,
    const ngraph::element::Type outPrecision,
    const bool addDequantizationAttribute) :
    _isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShapeIsDefined(false),
    addDequantizationAttribute(addDequantizationAttribute) {
}

DequantizationOperations::Subtract::Subtract(
    const std::vector<float>& values,
    const ngraph::element::Type outPrecision,
    const ngraph::Shape& constantShape,
    const bool addDequantizationAttribute,
    const size_t constantIndex) :
    _isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShape(constantShape),
    constantShapeIsDefined(true),
    addDequantizationAttribute(addDequantizationAttribute),
    constantIndex(constantIndex) {
}

bool DequantizationOperations::Subtract::empty() const noexcept {
    return _isEmpty;
}

DequantizationOperations::Subtract& DequantizationOperations::Subtract::setConstantPrecision(const ngraph::element::Type& precision) {
    constantPrecision = precision;
    return *this;
}

DequantizationOperations::Multiply::Multiply() :
    _isEmpty(true),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Multiply::Multiply(const float value, const bool addDequantizationAttribute) :
    _isEmpty(false),
    values({ value }),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false),
    addDequantizationAttribute(addDequantizationAttribute) {
}

DequantizationOperations::Multiply::Multiply(const std::vector<float>& values, const bool addDequantizationAttribute) :
    _isEmpty(values.empty()),
    values(values),
    outPrecision(ngraph::element::undefined),
    constantShapeIsDefined(false),
    addDequantizationAttribute(addDequantizationAttribute) {
}

DequantizationOperations::Multiply::Multiply(const std::vector<float>& values,
                                             const ngraph::element::Type outPrecision,
                                             const bool addDequantizationAttribute) :
    _isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShapeIsDefined(false),
    addDequantizationAttribute(addDequantizationAttribute) {
}

DequantizationOperations::Multiply::Multiply(
    const std::vector<float>& values,
    const ngraph::element::Type outPrecision,
    const ngraph::Shape& constantShape,
    const bool addDequantizationAttribute,
    const size_t constantIndex) :
    _isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShape(constantShape),
    addDequantizationAttribute(addDequantizationAttribute),
    constantIndex(constantIndex),
    constantShapeIsDefined(true) {
}

bool DequantizationOperations::Multiply::empty() const noexcept {
    return _isEmpty;
}

DequantizationOperations::Multiply& DequantizationOperations::Multiply::setConstantPrecision(const ngraph::element::Type& precision) {
    constantPrecision = precision;
    return *this;
}

DequantizationOperations::DequantizationOperations() {}

DequantizationOperations::DequantizationOperations(
    const Convert& convert,
    const Subtract& subtract,
    const Multiply& multiply) :
    convert(convert),
    subtract(subtract),
    multiply(multiply)
{}

bool DequantizationOperations::empty() const {
    return convert.empty() && subtract.empty() && multiply.empty();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
