// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_relu6(NodeContext& context) {
    auto x = context.get_input(0);
    return {context.mark_node(std::make_shared<opset8::Clamp>(x, 0., 6.))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov