// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>

namespace ngraph {
namespace builder {
namespace subgraph {

class Constant {
public:
    Constant();
    Constant(const float value);
    Constant(const std::vector<float>& values);
    Constant(const std::vector<float>& values, const ngraph::element::Type outPrecision);
    Constant(const std::vector<float>& values, const ngraph::element::Type outPrecision, const ngraph::Shape& shape);
    bool empty() const noexcept;

    std::vector<float> values;
    ngraph::element::Type outPrecision;
    ngraph::Shape shape;
    bool shapeIsDefined;
private:
    bool isEmpty;
};

inline std::ostream& operator<<(std::ostream& out, const Constant& constant) {
    return out << "_" << constant.values << "_" << constant.outPrecision << "_" << constant.shape;
}

}  // namespace subgraph
}  // namespace builder
}  // namespace ngraph
