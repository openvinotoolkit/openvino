// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "openvino/core/deprecated.hpp"
OPENVINO_SUPPRESS_DEPRECATED_START

#include "openvino/op/gather_elements.hpp"

namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
inline OutputVector gather_elements(const Node& node) {
    OutputVector ng_inputs{node.get_ng_inputs()};
    auto data = ng_inputs.at(0);
    auto indices = ng_inputs.at(1);
    auto axis = node.get_attribute_value<int64_t>("axis", 0);

    return {std::make_shared<ov::op::v6::GatherElements>(data, indices, axis)};
}
}  // namespace set_1
}  // namespace op
}  // namespace onnx_import
}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
