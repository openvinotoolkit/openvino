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

OutputVector translate_hardtanh(NodeContext& context) {
    float min = -1;
    float max = 1;
    if (!context.input_is_none(1)) {
        min = context.const_input<float>(1);
    }
    if (!context.input_is_none(2)) {
        max = context.const_input<float>(2);
    }
    return {context.mark_node(std::make_shared<opset10::Clamp>(context.get_input(0), min, max))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov