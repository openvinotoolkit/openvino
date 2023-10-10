// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/adaptive_avg_pool.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/tile.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector translate_adaptive_avg_pool_base(const NodeContext& context,
                                              const Output<Node>& tile_shape,
                                              const Output<Node>& slice_end) {
    num_inputs_check(context, 2, 2);
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));

    auto input_tensor = context.get_input(0);
    auto given_shape = context.get_input(1);

    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input_tensor, element::i32));
    auto shape_begin =
        context.mark_node(std::make_shared<v8::Slice>(input_shape, const_0, slice_end, const_1, const_0));
    auto output_shape = context.mark_node(std::make_shared<v0::Concat>(OutputVector{shape_begin, given_shape}, 0));

    auto tile = context.mark_node(std::make_shared<v0::Tile>(input_tensor, tile_shape));
    auto adaptive_avg_pool = context.mark_node(std::make_shared<v8::AdaptiveAvgPool>(tile, given_shape));
    auto reshape = context.mark_node(std::make_shared<v1::Reshape>(adaptive_avg_pool, output_shape, false));

    return {reshape};
};
}  // namespace

OutputVector translate_adaptive_avg_pool3d(const NodeContext& context) {
    auto const_tile_params = context.mark_node(v0::Constant::create(element::i32, Shape{5}, {1, 1, 1, 1, 1}));
    auto const_neg_3 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-3}));
    return translate_adaptive_avg_pool_base(context, const_tile_params, const_neg_3);
};

OutputVector translate_adaptive_avg_pool2d(const NodeContext& context) {
    auto const_tile_params = context.mark_node(v0::Constant::create(element::i32, Shape{4}, {1, 1, 1, 1}));
    auto const_neg_2 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-2}));
    return translate_adaptive_avg_pool_base(context, const_tile_params, const_neg_2);
};

OutputVector translate_adaptive_avg_pool1d(const NodeContext& context) {
    auto const_tile_params = context.mark_node(v0::Constant::create(element::i32, Shape{3}, {1, 1, 1}));
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    return translate_adaptive_avg_pool_base(context, const_tile_params, const_neg_1);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov