// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/shape_of.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_masked_fill(NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto data = context.get_input(0);
    auto mask = context.get_input(1);
    auto value = context.const_input<float>(2);
    auto data_shape = context.mark_node(std::make_shared<v3::ShapeOf>(data));
    auto value_const = context.mark_node(v0::Constant::create(element::f32, Shape({}), {value}));
    auto broadcasted_value = context.mark_node(std::make_shared<v3::Broadcast>(value_const, data_shape));
    auto bool_mask = context.mark_node(std::make_shared<v0::Convert>(mask, element::boolean));
    return {context.mark_node(std::make_shared<v1::Select>(bool_mask, broadcasted_value, data))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov