// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_adaptive_avg_pool3d(NodeContext& context) {
    auto const_tile_params = context.mark_node(opset10::Constant::create(element::i32, Shape{5}, {1, 1, 1, 1, 1}));
    auto const_0 = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {0}));
    auto const_1 = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {1}));
    auto const_neg_3 = context.mark_node(opset10::Constant::create(element::i32, Shape{1}, {-3}));

    auto input_tensor = context.get_input(0);
    auto given_shape = context.get_input(1);

    auto input_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(input_tensor, element::i32));
    auto shape_begin =
        context.mark_node(std::make_shared<opset10::Slice>(input_shape, const_0, const_neg_3, const_1, const_0));
    auto output_shape = context.mark_node(std::make_shared<opset10::Concat>(OutputVector{shape_begin, given_shape}, 0));

    auto tile = context.mark_node(std::make_shared<opset10::Tile>(input_tensor, const_tile_params));
    auto adaptive_avg_pool = context.mark_node(std::make_shared<opset10::AdaptiveAvgPool>(tile, given_shape));
    auto reshape = context.mark_node(std::make_shared<opset10::Reshape>(adaptive_avg_pool, output_shape, false));

    return {reshape};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov