// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/not_equal.hpp"
#include "pt_framework_node.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_is_nonzero(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);

    auto zero_tensor = context.mark_node(v0::Constant::create(element::boolean, Shape{1}, {false}));

    zero_tensor = context.mark_node(std::make_shared<v1::ConvertLike>(zero_tensor, input));
    auto result = context.mark_node(std::make_shared<v1::NotEqual>(input, zero_tensor));

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
