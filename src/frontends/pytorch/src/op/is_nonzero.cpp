// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/not_equal.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_is_nonzero(const NodeContext& context) {
    num_inputs_check(context, 1, 1);
    auto input = context.get_input(0);

    // aten::is_nonzero expects numel == 1; reshape to 0D scalar and compare to zero.
    auto scalar_shape = context.mark_node(v0::Constant::create(element::i32, Shape{0}, {}));
    auto scalar_input = context.mark_node(std::make_shared<v1::Reshape>(input, scalar_shape, true))->output(0);
    auto zero_tensor = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    zero_tensor = context.mark_node(std::make_shared<v1::ConvertLike>(zero_tensor, scalar_input));
    auto result = context.mark_node(std::make_shared<v1::NotEqual>(scalar_input, zero_tensor));

    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
