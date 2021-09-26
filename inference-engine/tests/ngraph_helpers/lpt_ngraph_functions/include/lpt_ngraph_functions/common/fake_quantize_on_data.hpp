// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

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
        const size_t quantizationLevel,
        const ngraph::Shape& constantShape,
        const std::vector<float>& inputLowValues,
        const std::vector<float>& inputHighValues,
        const std::vector<float>& outputLowValues,
        const std::vector<float>& outputHighValues,
        const ngraph::element::Type outputPrecision = ngraph::element::undefined,
        const std::vector<std::shared_ptr<Variant>>& attributes = {});

    virtual ~FakeQuantizeOnData();

    bool isSigned() const;
    virtual bool empty() const;

    size_t quantizationLevel;
    ngraph::Shape constantShape;
    std::vector<float> inputLowValues;
    std::vector<float> inputHighValues;
    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    ngraph::element::Type outputPrecision;
    std::vector<std::shared_ptr<Variant>> attributes;
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
    return out <<  "_" << data.quantizationLevel << data.constantShape << "_" << data.inputLowValues << "_" << data.inputHighValues <<
        "_" << data.outputLowValues << "_" << data.outputHighValues << "_" <<
        (data.outputPrecision == ngraph::element::undefined ? "" : data.outputPrecision.get_type_name());
}

class FakeQuantizeOnDataWithConstant {
public:
    FakeQuantizeOnDataWithConstant();

    FakeQuantizeOnDataWithConstant(
        const size_t quantizationLevel,
        const std::vector<ngraph::Shape>& constantShapes,
        const std::vector<float>& inputLowValues,
        const std::vector<float>& inputHighValues,
        const std::vector<float>& outputLowValues,
        const std::vector<float>& outputHighValues,
        const ngraph::element::Type outputPrecision = ngraph::element::undefined,
        const std::vector<std::shared_ptr<Variant>>& attributes = {});

    virtual ~FakeQuantizeOnDataWithConstant();

    virtual bool empty() const;

    size_t quantizationLevel;
    std::vector<ngraph::Shape> constantShapes;
    std::vector<float> inputLowValues;
    std::vector<float> inputHighValues;
    std::vector<float> outputLowValues;
    std::vector<float> outputHighValues;
    ngraph::element::Type outputPrecision;
    std::vector<std::shared_ptr<Variant>> attributes;
};

inline std::ostream& operator<<(std::ostream& out, const FakeQuantizeOnDataWithConstant& data) {
    return out <<  "_" << data.quantizationLevel <<
        (data.constantShapes.empty() ? ngraph::Shape{} : data.constantShapes[0]) << "_" <<
        data.inputLowValues << "_" << data.inputHighValues << "_" <<
        data.outputLowValues << "_" << data.outputHighValues << "_" <<
        (data.outputPrecision == ngraph::element::undefined ? "" : data.outputPrecision.get_type_name());
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
