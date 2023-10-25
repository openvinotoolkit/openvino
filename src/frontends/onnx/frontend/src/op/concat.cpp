// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op/concat.hpp"

#include <cstdint>

#include "default_opset.hpp"
#include "utils/common.hpp"

OPENVINO_SUPPRESS_DEPRECATED_START
namespace ngraph {
namespace onnx_import {
namespace op {
namespace set_1 {
OutputVector concat(const Node& node) {
    OutputVector inputs{node.get_ng_inputs()};
    std::int64_t axis = node.get_attribute_value<std::int64_t>("axis");
    OutputVector valid_inputs;
    std::copy_if(inputs.begin(), inputs.end(), std::back_inserter(valid_inputs), [](ov::Output<ov::Node>& in) -> bool {
        return !common::is_failsafe_node(in.get_node_shared_ptr());
    });
    return {std::make_shared<default_opset::Concat>(valid_inputs, axis)};
}

}  // namespace set_1

}  // namespace op

}  // namespace onnx_import

}  // namespace ngraph
OPENVINO_SUPPRESS_DEPRECATED_END
