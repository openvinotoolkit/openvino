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

OutputVector translate_gelu(NodeContext& context) {
    auto x = context.get_input(0);
    auto approximate = context.const_input<std::string>(1);
    // TODO: Add support for "tanh" approximate
    FRONT_END_OP_CONVERSION_CHECK(approximate == "none", "Unsupported approximate for Gelu: ", approximate);
    return {context.mark_node(std::make_shared<opset10::Gelu>(x))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov