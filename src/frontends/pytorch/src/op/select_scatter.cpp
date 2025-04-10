// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "helper_ops/gather_assign.hpp"
#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_select_scatter_fx(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    auto data = context.get_input(0);
    auto updates = context.get_input(1);
    auto dim = context.get_input(2);
    auto index = context.get_input(3);
    return {context.mark_node(std::make_shared<GatherAssign>(data, updates, index, dim))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov