// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "openvino/core/shape.hpp"
#include "openvino/core/type/element_type.hpp"

namespace ov {
namespace builder {
namespace subgraph {

class FakeQuantizeOnData {
public:
    FakeQuantizeOnData();

    FakeQuantizeOnData(const uint64_t quantizationLevel,
                       const ov::Shape& constantShape,
                       const std::vector<float>& inputLowValues,
                       const std::vector<float>& inputHighValues,
                       const std::vector<float>& outputLowValues,
                       const std::vector<float>& outputHighValues,
                       const ov::element::Type outputPrecision = ov::element::dynamic,
                       const std::vector<ov::Any>& attributes = {});

    virtual ~FakeQuantizeOnData();

    bool isSigned() const;
    virtual bool empty() const;

    uint64_t quantizationLevel;
    ov::Shape constantShape;
    std::vector<float> inputLowValues;
    std::vector<float> inputHighValues;
    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    ov::element::Type outputPrecision;
    std::vector<ov::Any> attributes;
};

inline std::ostream& operator<<(std::ostream& os, const std::vector<float>& values) {
    os << "{ ";
    for (size_t i = 0; i < values.size(); ++i) {
        os << values[i];
        if (i != (values.size() - 1ul)) {
            os << ", ";
        }
    }
    os << " }";
    return os;
}

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeOnData& data) {
    if (data.empty()) {
        return out << "{}";
    }
    return out << "level=" << data.quantizationLevel << "_shape=" << data.constantShape
               << "_input_low=" << data.inputLowValues << "_input_high=" << data.inputHighValues
               << "_output_low=" << data.outputLowValues << "_output_high" << data.outputHighValues << "_precision="
               << (data.outputPrecision == ov::element::dynamic ? "" : data.outputPrecision.get_type_name());
}

class FakeQuantizeOnDataWithConstant {
public:
    FakeQuantizeOnDataWithConstant();

    FakeQuantizeOnDataWithConstant(const uint64_t quantizationLevel,
                                   const std::vector<ov::Shape>& constantShapes,
                                   const std::vector<float>& inputLowValues,
                                   const std::vector<float>& inputHighValues,
                                   const std::vector<float>& outputLowValues,
                                   const std::vector<float>& outputHighValues,
                                   const ov::element::Type outputPrecision = ov::element::dynamic,
                                   const std::vector<ov::Any>& attributes = {},
                                   const bool addConverts = false,
                                   const ov::element::Type constantPrecision = ov::element::dynamic);
    virtual ~FakeQuantizeOnDataWithConstant();

    virtual bool empty() const;

    FakeQuantizeOnDataWithConstant& setConstantPrecision(ov::element::Type constantPrecision) {
        this->constantPrecision = constantPrecision;
        return *this;
    }

    uint64_t quantizationLevel;
    std::vector<ov::Shape> constantShapes;
    std::vector<float> inputLowValues;
    std::vector<float> inputHighValues;
    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    ov::element::Type outputPrecision;
    std::vector<ov::Any> attributes;
    bool addConverts;
    ov::element::Type constantPrecision;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeOnDataWithConstant& data) {
    if (data.empty()) {
        return out << "{}";
    }
    return out << "level=" << data.quantizationLevel
               << "_shape=" << (data.constantShapes.empty() ? ov::Shape{} : data.constantShapes[0])
               << "_input_low=" << data.inputLowValues << "_input_high=" << data.inputHighValues
               << "_output_low=" << data.outputLowValues << "_output_high=" << data.outputHighValues << "_output_precision="
               << (data.outputPrecision == ov::element::dynamic ? "" : data.outputPrecision.get_type_name())
               << "_constant_precision=" << (data.constantPrecision == ov::element::dynamic ? "" : data.constantPrecision.get_type_name());
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ov
