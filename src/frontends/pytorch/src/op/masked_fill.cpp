// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_masked_fill(const NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto data = context.get_input(0);
    auto mask = context.get_input(1);
    auto value = context.get_input(2);
    ov::pass::NodeRegistry rg;
    auto res = masked_fill(rg, data, mask, value);
    context.mark_nodes(rg.get());
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
