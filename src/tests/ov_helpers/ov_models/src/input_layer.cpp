// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <memory>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "ov_models/builders.hpp"
#include "ov_models/utils/ov_helpers.hpp"

namespace ngraph {
namespace builder {

std::shared_ptr<ov::Node> makeInputLayer(const element::Type& type,
                                         ov::test::utils::InputLayerType inputType,
                                         const std::vector<size_t>& shape) {
    std::shared_ptr<ov::Node> input;
    switch (inputType) {
    case ov::test::utils::InputLayerType::CONSTANT: {
        input = ov::test::utils::deprecated::make_constant<float>(type, shape, {}, true);
        break;
    }
    case ov::test::utils::InputLayerType::PARAMETER: {
        input = std::make_shared<ov::op::v0::Parameter>(type, ov::Shape(shape));
        break;
    }
    default:
        throw std::runtime_error("Unsupported inputType");
    }
    return input;
}
}  // namespace builder
}  // namespace ngraph
