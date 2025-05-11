// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/transpose.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_nonzero(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto cond = context.get_input(0);
    auto non_zero = context.mark_node(std::make_shared<v3::NonZero>(cond));
    auto input_order = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
    return {context.mark_node(std::make_shared<v1::Transpose>(non_zero, input_order))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov