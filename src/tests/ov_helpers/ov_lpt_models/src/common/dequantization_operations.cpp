// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/common/dequantization_operations.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace builder {
namespace subgraph {

DequantizationOperations::Convert::Convert() :
    isEmpty(true),
    outPrecision(ov::element::undefined)
{}

DequantizationOperations::Convert::Convert(const ov::element::Type outPrecision, const bool toRemove) :
    isEmpty(false),
    outPrecision(outPrecision)
{}

bool DequantizationOperations::Convert::empty() const noexcept {
    return isEmpty;
}

bool DequantizationOperations::Convert::equal(const DequantizationOperations::Convert& value) const noexcept {
    return (this->outPrecision == value.outPrecision) && (this->addDequantizationAttribute == value.addDequantizationAttribute);
}

DequantizationOperations::Subtract::Subtract() :
    isEmpty(true),
    outPrecision(ov::element::undefined),
    constantShapeIsDefined(false)
{}

DequantizationOperations::Subtract::Subtract(const float value, const bool toRemove) :
    isEmpty(false),
    values({ value }),
    outPrecision(ov::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Subtract::Subtract(const std::vector<float>& values) :
    isEmpty(values.empty()),
    values(values),
    outPrecision(ov::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Subtract::Subtract(
    const std::vector<float>& values,
    const ov::element::Type outPrecision) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Subtract::Subtract(
    const std::vector<float>& values,
    const ov::element::Type outPrecision,
    const ov::Shape& constantShape,
    const bool toRemove,
    const size_t constantIndex,
    const ov::element::Type constantPrecision,
    const bool addConvert,
    const ov::Node::RTMap& attributes,
    const ov::Node::RTMap& convertAttributes) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShape(constantShape),
    constantShapeIsDefined(true),
    constantIndex(constantIndex),
    constantPrecision(constantPrecision),
    addConvert(addConvert),
    attributes(attributes),
    convertAttributes(convertAttributes) {
}

bool DequantizationOperations::Subtract::empty() const noexcept {
    return isEmpty;
}

bool DequantizationOperations::Subtract::equal(const DequantizationOperations::Subtract& value) const noexcept {
    return
        (values == value.values) &&
        (outPrecision == value.outPrecision) &&
        (constantShape == value.constantShape) &&
        (constantShapeIsDefined == value.constantShapeIsDefined) &&
        (constantIndex == value.constantIndex);
}

DequantizationOperations::Subtract& DequantizationOperations::Subtract::setConstantPrecision(const ov::element::Type& precision) {
    constantPrecision = precision;
    return *this;
}

DequantizationOperations::Subtract& DequantizationOperations::Subtract::setAddConvert(bool value) {
    addConvert = value;
    return *this;
}

DequantizationOperations::Multiply::Multiply() :
    isEmpty(true),
    outPrecision(ov::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Multiply::Multiply(const float value) :
    isEmpty(false),
    values({ value }),
    outPrecision(ov::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Multiply::Multiply(const std::vector<float>& values) :
    isEmpty(values.empty()),
    values(values),
    outPrecision(ov::element::undefined),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Multiply::Multiply(const std::vector<float>& values, const ov::element::Type outPrecision) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShapeIsDefined(false) {
}

DequantizationOperations::Multiply::Multiply(
    const std::vector<float>& values,
    const ov::element::Type outPrecision,
    const ov::Shape& constantShape,
    const bool toRemove,
    const size_t constantIndex,
    ov::element::Type constantPrecision,
    const bool addConvert) :
    isEmpty(false),
    values(values),
    outPrecision(outPrecision),
    constantShape(constantShape),
    constantIndex(constantIndex),
    constantPrecision(constantPrecision),
    addConvert(addConvert),
    constantShapeIsDefined(true) {
}

bool DequantizationOperations::Multiply::empty() const noexcept {
    return isEmpty;
}

bool DequantizationOperations::Multiply::equal(const DequantizationOperations::Multiply& value) const noexcept {
    return
        (values == value.values) &&
        (outPrecision == value.outPrecision) &&
        (constantShape == value.constantShape) &&
        (constantIndex == value.constantIndex) &&
        (constantPrecision == value.constantPrecision) &&
        (constantShapeIsDefined == value.constantShapeIsDefined);
}

bool DequantizationOperations::equal(const DequantizationOperations& value) const noexcept {
    return
        (empty() == value.empty()) &&
        (convert == value.convert) &&
        (subtract == value.subtract) &&
        (multiply == value.multiply);
}

DequantizationOperations::Multiply& DequantizationOperations::Multiply::setConstantPrecision(const ov::element::Type& precision) {
    constantPrecision = precision;
    return *this;
}

DequantizationOperations::Multiply& DequantizationOperations::Multiply::setAddConvert(bool value) {
    addConvert = value;
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

void DequantizationOperations::setPrecision(const ov::element::Type& type) noexcept {
    convert.outPrecision = type;
    if (!subtract.addConvert)
        subtract.constantPrecision = type;
    subtract.outPrecision = type;
    if (!multiply.addConvert)
        multiply.constantPrecision = type;
    multiply.outPrecision = type;
}

bool DequantizationOperations::empty() const noexcept {
    return convert.empty() && subtract.empty() && multiply.empty();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
