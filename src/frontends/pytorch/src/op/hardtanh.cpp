// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/clamp.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_hardtanh(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    float min = -1;
    float max = 1;
    if (!context.input_is_none(1)) {
        min = context.const_input<float>(1);
    }
    if (!context.input_is_none(2)) {
        max = context.const_input<float>(2);
    }
    return {context.mark_node(std::make_shared<ov::op::v0::Clamp>(context.get_input(0), min, max))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov