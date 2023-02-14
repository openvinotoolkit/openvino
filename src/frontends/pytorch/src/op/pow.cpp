// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/power.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_pow(NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto lhs = context.get_input(0);
    auto rhs = context.get_input(1);
    align_eltwise_input_types(context, lhs, rhs, true);
    return {context.mark_node(std::make_shared<ov::op::v1::Power>(lhs, rhs))};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
