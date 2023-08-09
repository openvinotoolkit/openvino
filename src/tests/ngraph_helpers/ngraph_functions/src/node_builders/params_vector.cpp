// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>

#include "ngraph_functions/node_builders/param_vector.hpp"

namespace ov {
namespace test {
namespace utils {
namespace builder {

ov::ParameterVector make_params(const element::Type &type, const std::vector<std::vector<size_t>> &shapes) {
    ov::ParameterVector outs;
    for (const auto &shape : shapes) {
        auto paramNode = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(shape));
        outs.push_back(paramNode);
    }

    return outs;
}

ov::ParameterVector make_params(const element::Type &type, const std::vector<std::pair<std::string, std::vector<size_t>>> &inputs) {
    ov::ParameterVector outs;
    for (const auto &input : inputs) {
        const auto &name = input.first;
        const auto &shape = input.second;
        auto paramNode = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(shape));
        paramNode->set_friendly_name(name);
        outs.push_back(paramNode);
    }

    return outs;
}

ov::ParameterVector make_dynamic_params(const element::Type &type, const std::vector<ov::PartialShape> &shapes) {
    ov::ParameterVector outs;
    for (const auto &shape : shapes) {
        auto paramNode = std::make_shared<ov::op::v0::Parameter>(type, shape);
        outs.push_back(paramNode);
    }

    return outs;
}

ov::ParameterVector make_dynamic_params(const std::vector<element::Type>& types, const std::vector<ov::PartialShape>& shapes) {
    ov::ParameterVector outs;
    OPENVINO_ASSERT(types.size() == shapes.size());
    for (size_t i = 0; i < types.size(); i++) {
        auto paramNode = std::make_shared<ov::op::v0::Parameter>(types[i], shapes[i]);
        outs.push_back(paramNode);
    }
    return outs;
}

}  // namespace builder
}  // namespace utils
}  // namespace test
}  // namespace ov