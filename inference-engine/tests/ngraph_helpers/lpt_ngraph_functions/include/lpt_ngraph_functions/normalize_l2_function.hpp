// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>

#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class NormalizeL2ActualValues {
public:
    ngraph::element::Type precision;
    std::vector<int64_t> axes;
    std::vector<float> subtractValues;
    std::vector<float> mutliplyValues;
};

inline std::ostream& operator<<(std::ostream& out, const NormalizeL2ActualValues& values) {
    return out <<
        "_" << values.precision << "_" << values.axes.size() <<
        "_subtract" << values.subtractValues.size() <<
        "_mutliply" << values.mutliplyValues.size();
}

class NormalizeL2ExpectedValues {
public:
    ngraph::element::Type precision;
    std::vector<int64_t> axes;
    std::vector<float> subtractValues;
    std::vector<float> mutliplyValues;
};

inline std::ostream& operator<<(std::ostream& out, const NormalizeL2ExpectedValues& values) {
    return out <<
        "_" << values.precision << "_" << values.axes.size() <<
        "_subtract" << values.subtractValues.size() <<
        "_mutliply" << values.mutliplyValues.size();
}

class NormalizeL2Function {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const std::pair<ngraph::Shape, ngraph::Shape>& shapes,
        const ngraph::element::Type precisionOnActivation,
        const std::vector<uint64_t>& axes,
        const bool fuseMultiply,
        const bool shift);

    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& shape,
        const ngraph::op::EpsMode& epsMode,
        const NormalizeL2ActualValues& actualValues);

    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& shape,
        const ngraph::op::EpsMode& epsMode,
        const NormalizeL2ExpectedValues& expectedValues);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
