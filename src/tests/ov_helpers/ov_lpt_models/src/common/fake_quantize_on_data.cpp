// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ov_lpt_models/common/fake_quantize_on_data.hpp"
#include "openvino/opsets/opset1.hpp"

namespace ov {
namespace builder {
namespace subgraph {

FakeQuantizeOnData::FakeQuantizeOnData() : quantizationLevel(0) {}

FakeQuantizeOnData::FakeQuantizeOnData(
    const uint64_t quantizationLevel,
    const ov::Shape& constantShape,
    const std::vector<float>& inputLowValues,
    const std::vector<float>& inputHighValues,
    const std::vector<float>& outputLowValues,
    const std::vector<float>& outputHighValues,
    const ov::element::Type outputPrecision,
    const std::vector<ov::Any>& attributes) :
    quantizationLevel(quantizationLevel),
    constantShape(constantShape),
    inputLowValues(inputLowValues),
    inputHighValues(inputHighValues),
    outputLowValues(outputLowValues),
    outputHighValues(outputHighValues),
    outputPrecision(outputPrecision),
    attributes(attributes)
{}

FakeQuantizeOnData::~FakeQuantizeOnData() {}

bool FakeQuantizeOnData::isSigned() const {
    return std::any_of(outputLowValues.begin(), outputLowValues.end(), [](const float value) { return value < 0.f; }) ||
        std::any_of(outputHighValues.begin(), outputHighValues.end(), [](const float value) { return value < 0.f; });
}

bool FakeQuantizeOnData::empty() const {
    return (quantizationLevel == 0ul) &&
        constantShape.empty() &&
        inputLowValues.empty() &&
        inputHighValues.empty() &&
        outputLowValues.empty() &&
        outputHighValues.empty();
}

FakeQuantizeOnDataWithConstant::FakeQuantizeOnDataWithConstant()
    : quantizationLevel(0),
      outputPrecision(ov::element::dynamic) {}

FakeQuantizeOnDataWithConstant::FakeQuantizeOnDataWithConstant(
    const uint64_t quantizationLevel,
    const std::vector<ov::Shape>& constantShapes,
    const std::vector<float>& inputLowValues,
    const std::vector<float>& inputHighValues,
    const std::vector<float>& outputLowValues,
    const std::vector<float>& outputHighValues,
    const ov::element::Type outputPrecision,
    const std::vector<ov::Any>& attributes,
    const bool addConverts,
    const ov::element::Type constantPrecision) :
    quantizationLevel(quantizationLevel),
    constantShapes(constantShapes),
    inputLowValues(inputLowValues),
    inputHighValues(inputHighValues),
    outputLowValues(outputLowValues),
    outputHighValues(outputHighValues),
    outputPrecision(outputPrecision),
    attributes(attributes),
    addConverts(addConverts),
    constantPrecision(constantPrecision) {}

FakeQuantizeOnDataWithConstant::~FakeQuantizeOnDataWithConstant() {}

bool FakeQuantizeOnDataWithConstant::empty() const {
    return (quantizationLevel == 0ul) &&
        constantShapes.empty() &&
        inputLowValues.empty() &&
        inputHighValues.empty() &&
        outputLowValues.empty() &&
        outputHighValues.empty();
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
