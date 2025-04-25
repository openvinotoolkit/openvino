// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/floor_mod.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_remainder(const NodeContext& context) {
    num_inputs_check(context, 2, 2);
    Output<Node> x;
    Output<Node> y;
    std::tie(x, y) = get_inputs_with_promoted_types(context, 0, 1);
    return {context.mark_node(std::make_shared<v1::FloorMod>(x, y))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
