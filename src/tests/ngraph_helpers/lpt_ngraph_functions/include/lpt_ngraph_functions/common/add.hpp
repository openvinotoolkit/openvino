// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class Add {
public:
    Add();
    Add(const float value);
    Add(const std::vector<float>& values);
    Add(const std::vector<float>& values, const ngraph::element::Type outPrecision);
    Add(const std::vector<float>& values, const ngraph::element::Type outPrecision, const ngraph::Shape& constantShape);
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
