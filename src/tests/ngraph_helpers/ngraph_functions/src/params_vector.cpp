// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vector>
#include <memory>

#include "ngraph_functions/builders.hpp"
#include "openvino/core/partial_shape.hpp"

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

ngraph::ParameterVector makeParams(const element::Type &type, const std::vector<std::pair<std::string, std::vector<size_t>>> &inputs) {
    ngraph::ParameterVector outs;
    for (const auto &input : inputs) {
        const auto &name = input.first;
        const auto &shape = input.second;
        auto paramNode = std::make_shared<ngraph::opset1::Parameter>(type, ngraph::Shape(shape));
        paramNode->set_friendly_name(name);
        outs.push_back(paramNode);
    }

    return outs;
}

ngraph::ParameterVector makeDynamicParams(const element::Type &type, const std::vector<ov::PartialShape> &shapes) {
    ngraph::ParameterVector outs;
    for (const auto &shape : shapes) {
        auto paramNode = std::make_shared<ngraph::opset1::Parameter>(type, shape);
        outs.push_back(paramNode);
    }

    return outs;
}

ngraph::ParameterVector makeDynamicParams(const std::vector<element::Type>& types, const std::vector<ov::PartialShape>& shapes) {
    ngraph::ParameterVector outs;
    NGRAPH_CHECK(types.size() == shapes.size());
    for (size_t i = 0; i < types.size(); i++) {
        auto paramNode = std::make_shared<ov::op::v0::Parameter>(types[i], shapes[i]);
        outs.push_back(paramNode);
    }
    return outs;
}

}  // namespace builder
}  // namespace ngraph