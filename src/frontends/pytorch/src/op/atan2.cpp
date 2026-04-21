// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"

#include "openvino/op/atan2.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_atan2(const NodeContext& context) {
    // aten::atan2(Tensor self, Tensor other) → Tensor
    num_inputs_check(context, 2, 2);
    Output<Node> lhs;
    Output<Node> rhs;
    std::tie(lhs, rhs) = get_inputs_with_promoted_types(context, 0, 1);
    return {context.mark_node(std::make_shared<v17::Atan2>(lhs, rhs))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
