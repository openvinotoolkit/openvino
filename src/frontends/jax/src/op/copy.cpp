// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/core/node.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/frontend/jax/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace jax {
namespace op {

using namespace ov::op;

OutputVector translate_copy(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    Output<Node> src = context.get_input(0);
    auto res = src.get_node()->clone_with_new_inputs(src.get_node()->input_values());
    return {res};
};

}  // namespace op
}  // namespace jax
}  // namespace frontend
}  // namespace ov
