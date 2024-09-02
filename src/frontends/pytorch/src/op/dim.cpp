// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_dim(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    Output<Node> rank = std::get<1>(get_shape_rank(context, context.get_input(0), true));
    return {rank};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov