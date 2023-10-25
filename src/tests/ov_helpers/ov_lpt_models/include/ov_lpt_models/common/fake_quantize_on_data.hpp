// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <vector>
#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class FakeQuantizeOnData {
public:
    FakeQuantizeOnData();

    FakeQuantizeOnData(
        const uint64_t quantizationLevel,
        const ngraph::Shape& constantShape,
        const std::vector<float>& inputLowValues,
        const std::vector<float>& inputHighValues,
        const std::vector<float>& outputLowValues,
        const std::vector<float>& outputHighValues,
        const ngraph::element::Type outputPrecision = ngraph::element::undefined,
        const std::vector<ov::Any>& attributes = {});

    virtual ~FakeQuantizeOnData();

    bool isSigned() const;
    virtual bool empty() const;

    uint64_t quantizationLevel;
    ngraph::Shape constantShape;
    std::vector<float> inputLowValues;
    std::vector<float> inputHighValues;
    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    ngraph::element::Type outputPrecision;
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
    return out << "level=" << data.quantizationLevel <<
        "_shape=" << data.constantShape <<
        "_input_low=" << data.inputLowValues <<
        "_input_high=" << data.inputHighValues <<
        "_output_low=" << data.outputLowValues <<
        "_output_high" << data.outputHighValues <<
        "_precision=" << (data.outputPrecision == ngraph::element::undefined ? "" : data.outputPrecision.get_type_name());
}

class FakeQuantizeOnDataWithConstant {
public:
    FakeQuantizeOnDataWithConstant();

    FakeQuantizeOnDataWithConstant(
        const uint64_t quantizationLevel,
        const std::vector<ngraph::Shape>& constantShapes,
        const std::vector<float>& inputLowValues,
        const std::vector<float>& inputHighValues,
        const std::vector<float>& outputLowValues,
        const std::vector<float>& outputHighValues,
        const ngraph::element::Type outputPrecision = ngraph::element::undefined,
        const std::vector<ov::Any>& attributes = {},
        const bool addConverts = false);
    virtual ~FakeQuantizeOnDataWithConstant();

    virtual bool empty() const;

    uint64_t quantizationLevel;
    std::vector<ngraph::Shape> constantShapes;
    std::vector<float> inputLowValues;
    std::vector<float> inputHighValues;
    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    ngraph::element::Type outputPrecision;
    std::vector<ov::Any> attributes;
    bool addConverts;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeOnDataWithConstant& data) {
    if (data.empty()) {
        return out << "{}";
    }
    return out << "level=" << data.quantizationLevel <<
        "_shape=" <<(data.constantShapes.empty() ? ngraph::Shape{} : data.constantShapes[0]) <<
        "_input_low=" << data.inputLowValues <<
        "_input_high=" << data.inputHighValues <<
        "_output_low=" << data.outputLowValues <<
        "_output_high=" << data.outputHighValues <<
        "_precision=" << (data.outputPrecision == ngraph::element::undefined ? "" : data.outputPrecision.get_type_name());
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
