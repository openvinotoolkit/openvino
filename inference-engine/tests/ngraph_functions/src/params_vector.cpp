// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"

namespace ngraph {
namespace builder {
ngraph::ParameterVector makeParams(const element::Type &type, const std::vector<std::vector<size_t>> &shapes) {
    ngraph::ParameterVector outs;
    for (const auto &shape : shapes) {
        auto paramNode = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(shape));
        outs.push_back(paramNode);
    }
    return outs;
}
}  // namespace builder
}  // namespace ngraph