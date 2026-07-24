// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/reshape.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_take(const NodeContext& context) {
    // aten::take(Tensor self, Tensor index) -> Tensor
    // aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 2, 3);
    auto x = context.get_input(0);
    auto index = context.get_input(1);
    index = context.mark_node(std::make_shared<ov::op::v0::Convert>(index, element::i32));
    // Flatten the input tensor to 1D
    auto minus_1 = context.mark_node(ov::op::v0::Constant::create(element::i32, Shape{1}, {-1}));
    x = context.mark_node(std::make_shared<ov::op::v1::Reshape>(x, minus_1, false));
    // Use Gather to select elements from the flattened input along axis 0
    auto axis_0 = context.mark_node(ov::op::v0::Constant::create(element::i64, Shape{}, {0}));
    auto gather = context.mark_node(std::make_shared<ov::op::v8::Gather>(x, index, axis_0));
    if (!context.input_is_none(2)) {
        context.mutate_input(2, gather);
    }
    return {gather};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
