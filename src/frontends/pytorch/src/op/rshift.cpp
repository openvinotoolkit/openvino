// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/bitwise_right_shift.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_rshift(const NodeContext& context) {
    num_inputs_check(context, 2, 2);

    auto input_tensor = context.get_input(0);
    auto shift_amount = context.get_input(1);

    if (input_tensor.get_element_type() != shift_amount.get_element_type()) {
        shift_amount = context.mark_node(std::make_shared<v1::ConvertLike>(shift_amount, input_tensor));
    }

    auto rshift_node = context.mark_node(std::make_shared<v15::BitwiseRightShift>(input_tensor, shift_amount));
    return {rshift_node};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
