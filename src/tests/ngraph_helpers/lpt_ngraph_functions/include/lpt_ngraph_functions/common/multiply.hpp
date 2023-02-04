// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class Multiply {
public:
    Multiply();
    Multiply(const float value);
    Multiply(const std::vector<float>& values);
    Multiply(const std::vector<float>& values, const ngraph::element::Type outPrecision);
    Multiply(const std::vector<float>& values, const ngraph::element::Type outPrecision, const ngraph::Shape& constantShape);
    bool empty() const noexcept;

    std::vector<float> values;
    ngraph::element::Type outPrecision;
    ngraph::Shape constantShape;
    bool constantShapeIsDefined;
private:
    bool isEmpty;
};

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
