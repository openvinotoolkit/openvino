// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <vector>
#include <string>
#include <map>

#include <ngraph/ngraph.hpp>
#include <ngraph/opsets/opset1.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class DepthToSpaceActualValues {
public:
    ngraph::element::Type precision;
    std::vector<float> subtractValues;
    std::vector<float> mutliplyValues;
};

inline std::ostream& operator<<(std::ostream& out, const DepthToSpaceActualValues& values) {
    return out <<
        "_" << values.precision <<
        "_subtract" << values.subtractValues.size() <<
        "_mutliply" << values.mutliplyValues.size();
}

class DepthToSpaceExpectedValues {
public:
    ngraph::element::Type precision;
    std::vector<float> subtractValues;
    std::vector<float> mutliplyValues;
};

inline std::ostream& operator<<(std::ostream& out, const DepthToSpaceExpectedValues& values) {
    return out <<
        "_" << values.precision <<
        "_subtract" << values.subtractValues.size() <<
        "_mutliply" << values.mutliplyValues.size();
}


class DepthToSpaceFunction {
public:
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
        const size_t blockSize);
    static std::shared_ptr<ngraph::Function> getOriginal(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
        const size_t blockSize,
        const DepthToSpaceActualValues& actualValues);
    static std::shared_ptr<ngraph::Function> getReference(
        const ngraph::element::Type precision,
        const ngraph::Shape& inputShape,
        const ngraph::opset1::DepthToSpace::DepthToSpaceMode mode,
        const size_t blockSize,
        const DepthToSpaceExpectedValues& expectedValues);
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
