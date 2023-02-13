// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_squeeze(NodeContext& context) {
    auto inputs = context.inputs();
    FRONT_END_OP_CONVERSION_CHECK(inputs.size() >= 1, "Operation has no inputs.");
    FRONT_END_OP_CONVERSION_CHECK(!context.input_is_none(0), "Input should not be None.");
    if (inputs.size() == 1 || context.input_is_none(1)) {
        return {context.mark_node(std::make_shared<opset10::Squeeze>(inputs[0]))};
    }
    return {context.mark_node(std::make_shared<opset10::Squeeze>(inputs[0], inputs[1]))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
