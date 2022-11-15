// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_reshape(NodeContext& context) {
    const auto arg = context.get_input(0);
    const auto shape_pattern = context.get_input(1);
    return {context.mark_node(std::make_shared<opset8::Reshape>(arg, shape_pattern, false))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
