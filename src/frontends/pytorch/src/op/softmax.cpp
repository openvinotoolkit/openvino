// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_softmax(NodeContext& context) {
    auto x = context.get_input(0);
    auto axis = context.const_input<int64_t>(1);
    return {context.mark_node(std::make_shared<opset10::Softmax>(x, axis))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov