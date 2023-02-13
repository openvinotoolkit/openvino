// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/tile.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_repeat(NodeContext& context) {
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto repeats = context.get_input(1);
    auto one = context.mark_node(v0::Constant::create(element::i64, Shape{}, {1}));
    auto sizes_shape = context.mark_node(std::make_shared<v3::ShapeOf>(repeats, element::i64));
    auto expand_shape = context.mark_node(std::make_shared<v3::Broadcast>(one, sizes_shape));
    auto expanded_input =
        context.mark_node(std::make_shared<v3::Broadcast>(x, expand_shape, BroadcastType::BIDIRECTIONAL));
    return {context.mark_node(std::make_shared<v0::Tile>(expanded_input, repeats))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov