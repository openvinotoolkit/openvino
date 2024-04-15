// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_set_item(const NodeContext& context) {
    // schema: aten::_set_item.t(t[](a!) l, int idx, t(b -> *) el) -> t[](a!)
    // _set_item inserts element in list
    num_inputs_check(context, 3, 3);
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto input = context.get_input(0);
    auto idx = context.get_input(1);
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto shape_0 = context.mark_node(std::make_shared<v8::Gather>(shape, zero, zero));
    shape_0 = context.mark_node(std::make_shared<v1::ConvertLike>(shape_0, idx));
    shape_0 = context.mark_node(std::make_shared<v0::Unsqueeze>(shape_0, zero));
    // ScatterUpdate-3 doesn't support negative indices
    idx = normalize_axis(context, idx, shape_0);
    auto value = context.get_input(2);
    auto value_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(value, zero));
    value_unsqueezed = context.mark_node(std::make_shared<v1::ConvertLike>(value_unsqueezed, input));
    auto res = context.mark_node(std::make_shared<v3::ScatterUpdate>(input, idx, value_unsqueezed, zero));
    context.mutate_input(0, res);
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov