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

using namespace ov::op;

OutputVector translate_take(const NodeContext& context) {
    // aten::take(Tensor self, Tensor index) -> Tensor
    // aten::take.out(Tensor self, Tensor index, *, Tensor(a!) out) -> Tensor(a!)
    num_inputs_check(context, 2, 3);
    auto self = context.get_input(0);
    auto index = context.get_input(1);

    index = context.mark_node(std::make_shared<v0::Convert>(index, element::i64));

    auto minus_one = context.mark_node(v0::Constant::create(element::i64, Shape{1}, {-1}));
    auto flattened = context.mark_node(std::make_shared<v1::Reshape>(self, minus_one, false));

    auto axis = context.mark_node(v0::Constant::create(element::i64, Shape{}, {0}));
    auto gather = context.mark_node(std::make_shared<v8::Gather>(flattened, index, axis));

    if (!context.input_is_none(2)) {
        context.mutate_input(2, gather);
    }
    return {gather};
}

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
