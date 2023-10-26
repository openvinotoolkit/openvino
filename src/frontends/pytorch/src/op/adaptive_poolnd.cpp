// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/adaptive_avg_pool.hpp"
#include "openvino/op/adaptive_max_pool.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
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

std::tuple<Output<Node>, Output<Node>> get_tile_input_and_output_shape(const NodeContext& context,
                                                                       const Output<Node>& input_tensor,
                                                                       const Output<Node>& given_shape,
                                                                       const Output<Node>& tile_shape,
                                                                       const Output<Node>& slice_end) {
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input_tensor, element::i32));
    auto shape_begin =
        context.mark_node(std::make_shared<v8::Slice>(input_shape, const_0, slice_end, const_1, const_0));
    Output<Node> output_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{shape_begin, given_shape}, 0));
    Output<Node> tile = context.mark_node(std::make_shared<v0::Tile>(input_tensor, tile_shape));
    return std::make_tuple(tile, output_shape);
};

OutputVector translate_adaptive_avg_pool_base(const NodeContext& context,
                                              const Output<Node>& tile_shape,
                                              const Output<Node>& slice_end) {
    num_inputs_check(context, 2, 2);

    auto input_tensor = context.get_input(0);
    auto given_shape = context.get_input(1);
    Output<Node> tile_input;
    Output<Node> output_shape;
    std::tie(tile_input, output_shape) =
        get_tile_input_and_output_shape(context, input_tensor, given_shape, tile_shape, slice_end);
    auto adaptive_avg_pool = context.mark_node(std::make_shared<v8::AdaptiveAvgPool>(tile_input, given_shape));
    auto reshape = context.mark_node(std::make_shared<v1::Reshape>(adaptive_avg_pool, output_shape, false));
    return {reshape};
};

OutputVector translate_adaptive_max_pool_base(const NodeContext& context,
                                              const Output<Node>& tile_shape,
                                              const Output<Node>& slice_end) {
    num_inputs_check(context, 2, 2);

    auto input_tensor = context.get_input(0);
    auto given_shape = context.get_input(1);
    Output<Node> tile_input;
    Output<Node> output_shape;
    std::tie(tile_input, output_shape) =
        get_tile_input_and_output_shape(context, input_tensor, given_shape, tile_shape, slice_end);

    auto adaptive_max_pool =
        context.mark_node(std::make_shared<v8::AdaptiveMaxPool>(tile_input, given_shape, element::i32));
    auto pooled_tensor = adaptive_max_pool->output(0);
    auto pooled_indices = adaptive_max_pool->output(1);
    // adaptive max pool in torch return indices in i64, indices_element_type i64 is not implented on ov runtime side
    pooled_indices = context.mark_node(std::make_shared<v0::Convert>(pooled_indices, element::i64));
    pooled_tensor = context.mark_node(std::make_shared<v1::Reshape>(pooled_tensor, output_shape, false));
    pooled_indices = context.mark_node(std::make_shared<v1::Reshape>(pooled_indices, output_shape, false));
    // aten::adaptive_max_pool{n}d always returns tuple with 2 tensors: pooled tensor and indicies
    // output selecting only first or preserve both made outside of operation by return_indices flag
    return {pooled_tensor, pooled_indices};
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

OutputVector translate_adaptive_max_pool3d(const NodeContext& context) {
    auto const_tile_params = context.mark_node(v0::Constant::create(element::i32, Shape{5}, {1, 1, 1, 1, 1}));
    auto const_neg_3 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-3}));
    return translate_adaptive_max_pool_base(context, const_tile_params, const_neg_3);
};

OutputVector translate_adaptive_max_pool2d(const NodeContext& context) {
    auto const_tile_params = context.mark_node(v0::Constant::create(element::i32, Shape{4}, {1, 1, 1, 1}));
    auto const_neg_2 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-2}));
    return translate_adaptive_max_pool_base(context, const_tile_params, const_neg_2);
};

OutputVector translate_adaptive_max_pool1d(const NodeContext& context) {
    auto const_tile_params = context.mark_node(v0::Constant::create(element::i32, Shape{3}, {1, 1, 1}));
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    return translate_adaptive_max_pool_base(context, const_tile_params, const_neg_1);
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov