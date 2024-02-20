// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/scatter_update.hpp"
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
    auto idx_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(idx, zero));
    auto value = context.get_input(2);
    auto value_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(value, zero));
    auto res = context.mark_node(std::make_shared<v3::ScatterUpdate>(input, idx_unsqueezed, value_unsqueezed, zero));
    context.mutate_input(0, res);
    return {res};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov