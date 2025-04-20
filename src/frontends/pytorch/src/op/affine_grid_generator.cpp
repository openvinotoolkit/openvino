// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/convert_like.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;
using namespace std;

namespace {

OutputVector translate_linspace_from_neg_one(const NodeContext& context,
                                             Output<Node> grid,
                                             int64_t num_steps,
                                             bool align_corners) {
    // aten::linspace_from_neg_one(Output<Node> grid, int64_t num_steps, bool align_corners) -> Tensor
    auto start = context.mark_node(v0::Constant::create(element::f32, Shape{}, {-1.0f}));

    auto steps = context.mark_node(std::make_shared<v0::Constant>(element::i64, Shape{}, &num_steps));
    steps = context.mark_node(std::make_shared<v0::Convert>(steps, element::f32));

    auto end = context.mark_node(v0::Constant::create(element::f32, Shape{}, {1.0f}));

    auto const_0 = v0::Constant::create(element::f32, Shape{}, {0});
    auto const_1 = v0::Constant::create(element::f32, Shape{}, {1});

    if (num_steps <= 1) {
        auto linspace = context.mark_node(v0::Constant::create(element::f32, Shape{}, {0}));
        linspace = context.mark_node(std::make_shared<v1::ConvertLike>(linspace, grid));
        return {linspace};
    }

    auto step_range = context.mark_node(std::make_shared<v4::Range>(const_0, steps, const_1, element::f32));

    auto sub_end_start = context.mark_node(std::make_shared<v1::Subtract>(end, start));
    auto sub_steps_1 = context.mark_node(std::make_shared<v1::Subtract>(steps, const_1));
    auto step_multiplier = context.mark_node(std::make_shared<v1::Divide>(sub_end_start, sub_steps_1));
    auto is_single_step = context.mark_node(std::make_shared<v1::Equal>(steps, const_1));
    auto select_multiplier = context.mark_node(std::make_shared<v1::Select>(is_single_step, const_0, step_multiplier));
    auto step_values = context.mark_node(std::make_shared<v1::Multiply>(step_range, select_multiplier));

    auto linspace = context.mark_node(std::make_shared<v1::Add>(step_values, start));

    if (!align_corners) {
        auto scale_factor = context.mark_node(std::make_shared<v1::Divide>(sub_steps_1, steps));
        linspace = context.mark_node(std::make_shared<v1::Multiply>(linspace, scale_factor));
    }

    linspace = context.mark_node(std::make_shared<v1::ConvertLike>(linspace, grid));

    return {linspace};
}

OutputVector translate_make_base_grid_4D(const NodeContext& context,
                                         Output<Node> theta,
                                         int64_t N,
                                         int64_t C,
                                         int64_t H,
                                         int64_t W,
                                         bool align_corners) {
    // aten::make_base_grid_4D(Tensor theta, int64_t N, int64_t C, int64_t H, int64_t W, bool align_corners) -> Tensor
    auto x_coords = translate_linspace_from_neg_one(context, theta, W, align_corners)[0];
    auto y_coords = translate_linspace_from_neg_one(context, theta, H, align_corners)[0];

    auto unsqueeze_axis =
        context.mark_node(std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1}.data()));
    auto y_coords_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(y_coords, unsqueeze_axis));

    auto hw_shape =
        context.mark_node(std::make_shared<v0::Constant>(element::i64, Shape{2}, std::vector<int64_t>{H, W}.data()));
    auto x_grid = context.mark_node(std::make_shared<v3::Broadcast>(x_coords, hw_shape, BroadcastType::NUMPY));
    auto y_grid =
        context.mark_node(std::make_shared<v3::Broadcast>(y_coords_unsqueezed, hw_shape, BroadcastType::NUMPY));

    auto ones =
        context.mark_node(std::make_shared<v0::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f}.data()));
    auto z_grid = context.mark_node(std::make_shared<v3::Broadcast>(ones, hw_shape, BroadcastType::NUMPY));
    z_grid = context.mark_node(std::make_shared<v1::ConvertLike>(z_grid, x_grid));

    auto reshape_axis =
        context.mark_node(std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1}.data()));
    auto x_reshaped = context.mark_node(std::make_shared<v0::Unsqueeze>(x_grid, reshape_axis));
    auto y_reshaped = context.mark_node(std::make_shared<v0::Unsqueeze>(y_grid, reshape_axis));
    auto z_reshaped = context.mark_node(std::make_shared<v0::Unsqueeze>(z_grid, reshape_axis));

    OutputVector grid_coords = {x_reshaped, y_reshaped, z_reshaped};
    auto grid_2d = context.mark_node(std::make_shared<v0::Concat>(grid_coords, -1));

    auto batch_shape = context.mark_node(
        std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{N, H, W, 3}.data()));
    auto grid_with_batch =
        context.mark_node(std::make_shared<v3::Broadcast>(grid_2d, batch_shape, BroadcastType::NUMPY));
    return {grid_with_batch};
}

OutputVector translate_affine_grid_generator_4D(const NodeContext& context,
                                                Output<Node> theta,
                                                int64_t N,
                                                int64_t C,
                                                int64_t H,
                                                int64_t W,
                                                bool align_corners) {
    auto base_grid = translate_make_base_grid_4D(context, theta, N, C, H, W, align_corners)[0];

    auto reshape_shape1 = context.mark_node(
        std::make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{N, H * W, 3}.data()));
    auto reshaped_grid = context.mark_node(std::make_shared<v1::Reshape>(base_grid, reshape_shape1, false));

    auto transpose_order =
        context.mark_node(std::make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{0, 2, 1}.data()));
    auto transposed_theta = context.mark_node(std::make_shared<v1::Transpose>(theta, transpose_order));

    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(reshaped_grid, transposed_theta));
    auto reshape_shape2 = context.mark_node(
        std::make_shared<v0::Constant>(element::i64, Shape{4}, std::vector<int64_t>{N, H, W, 2}.data()));
    auto final_grid = context.mark_node(std::make_shared<v1::Reshape>(matmul, reshape_shape2, false));

    return {final_grid};
}

OutputVector translate_make_base_grid_5D(const NodeContext& context,
                                         Output<Node> theta,
                                         int64_t N,
                                         int64_t C,
                                         int64_t D,
                                         int64_t H,
                                         int64_t W,
                                         bool align_corners) {
    // aten::make_base_grid_5D(Tensor theta, int64_t N, int64_t C, int64_t D, int64_t H, int64_t W, bool align_corners)
    // -> Tensor
    auto x_coords = translate_linspace_from_neg_one(context, theta, W, align_corners)[0];
    auto y_coords = translate_linspace_from_neg_one(context, theta, H, align_corners)[0];
    auto z_coords = translate_linspace_from_neg_one(context, theta, D, align_corners)[0];

    auto unsqueeze_axis =
        context.mark_node(std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1}.data()));
    auto y_coords_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(y_coords, unsqueeze_axis));
    auto z_coords_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(z_coords, unsqueeze_axis));
    auto z_coords_unsqueezed_twice =
        context.mark_node(std::make_shared<v0::Unsqueeze>(z_coords_unsqueezed, unsqueeze_axis));

    auto dhw_shape =
        context.mark_node(std::make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{D, H, W}.data()));
    auto x_grid = context.mark_node(std::make_shared<v3::Broadcast>(x_coords, dhw_shape, BroadcastType::NUMPY));
    auto y_grid =
        context.mark_node(std::make_shared<v3::Broadcast>(y_coords_unsqueezed, dhw_shape, BroadcastType::NUMPY));
    auto z_grid =
        context.mark_node(std::make_shared<v3::Broadcast>(z_coords_unsqueezed_twice, dhw_shape, BroadcastType::NUMPY));

    auto ones =
        context.mark_node(std::make_shared<v0::Constant>(element::f32, Shape{1}, std::vector<float>{1.0f}.data()));
    auto w_grid = context.mark_node(std::make_shared<v3::Broadcast>(ones, dhw_shape, BroadcastType::NUMPY));
    w_grid = context.mark_node(std::make_shared<v1::ConvertLike>(w_grid, x_grid));

    auto reshape_axis =
        context.mark_node(std::make_shared<v0::Constant>(element::i64, Shape{1}, std::vector<int64_t>{-1}.data()));
    auto x_reshaped = context.mark_node(std::make_shared<v0::Unsqueeze>(x_grid, reshape_axis));
    auto y_reshaped = context.mark_node(std::make_shared<v0::Unsqueeze>(y_grid, reshape_axis));
    auto z_reshaped = context.mark_node(std::make_shared<v0::Unsqueeze>(z_grid, reshape_axis));
    auto w_reshaped = context.mark_node(std::make_shared<v0::Unsqueeze>(w_grid, reshape_axis));

    OutputVector grid_coords = {x_reshaped, y_reshaped, z_reshaped, w_reshaped};
    auto grid_3d = context.mark_node(std::make_shared<v0::Concat>(grid_coords, -1));
    auto batch_shape = context.mark_node(
        std::make_shared<v0::Constant>(element::i64, Shape{5}, std::vector<int64_t>{N, D, H, W, 4}.data()));
    auto grid_with_batch =
        context.mark_node(std::make_shared<v3::Broadcast>(grid_3d, batch_shape, BroadcastType::NUMPY));
    return {grid_with_batch};
}

OutputVector translate_affine_grid_generator_5D(const NodeContext& context,
                                                Output<Node> theta,
                                                int64_t N,
                                                int64_t C,
                                                int64_t D,
                                                int64_t H,
                                                int64_t W,
                                                bool align_corners) {
    auto base_grid = translate_make_base_grid_5D(context, theta, N, C, D, H, W, align_corners)[0];

    auto reshape_shape1 = context.mark_node(
        std::make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{N, D * H * W, 4}.data()));
    auto reshaped_grid = context.mark_node(std::make_shared<v1::Reshape>(base_grid, reshape_shape1, false));

    auto transpose_order =
        context.mark_node(std::make_shared<v0::Constant>(element::i64, Shape{3}, std::vector<int64_t>{0, 2, 1}.data()));
    auto transposed_theta = context.mark_node(std::make_shared<v1::Transpose>(theta, transpose_order));

    auto matmul = context.mark_node(std::make_shared<v0::MatMul>(reshaped_grid, transposed_theta, false, false));
    auto reshape_shape2 = context.mark_node(
        std::make_shared<v0::Constant>(element::i64, Shape{5}, std::vector<int64_t>{N, D, H, W, 3}.data()));
    auto final_grid = context.mark_node(std::make_shared<v1::Reshape>(matmul, reshape_shape2, false));

    return {final_grid};
}

}  // namespace

OutputVector translate_affine_grid_generator(const NodeContext& context) {
    // aten::affine_grid_generator(Tensor theta, int64_t N, int64_t M, bool align_corners=False) -> Tensor
    num_inputs_check(context, 2, 3);
    auto theta = context.get_input(0);
    auto size = context.const_input<std::vector<int64_t>>(1);

    bool align_corners = false;
    if (!context.input_is_none(2)) {
        align_corners = context.const_input<int64_t>(2);
    }

    if (size.size() == 4) {
        return translate_affine_grid_generator_4D(context, theta, size[0], size[1], size[2], size[3], align_corners);
    } else {
        return translate_affine_grid_generator_5D(context,
                                                  theta,
                                                  size[0],
                                                  size[1],
                                                  size[2],
                                                  size[3],
                                                  size[4],
                                                  align_corners);
    }
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
