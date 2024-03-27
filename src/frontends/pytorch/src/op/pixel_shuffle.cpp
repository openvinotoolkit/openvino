// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_pixel_shuffle(const NodeContext& context) {
    // aten::pixel_shuffle(Tensor self, int upscale_factor) -> Tensor
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto upscale_factor = get_input_as_i32(context, 1);
    auto neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto neg_3 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-3}));
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto zero_s = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto one_s = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    Output<Node> shape;
    Output<Node> rank;
    std::tie(shape, rank) = get_shape_rank(context, x, true);
    // 1. Reshape input to [*, -1, r, r, H, W], where r is upscale factor
    auto indices = context.mark_node(v0::Constant::create(element::i32, Shape{3}, {-3, -2, -1}));
    auto dims = context.mark_node(std::make_shared<v8::Gather>(shape, indices, zero_s));
    auto dims_splitted = context.mark_node(std::make_shared<v1::Split>(dims, zero_s, 3));
    auto c = dims_splitted->output(0);
    auto h = dims_splitted->output(1);
    auto w = dims_splitted->output(2);
    auto dims_before = context.mark_node(std::make_shared<v8::Slice>(shape, zero, neg_3, one));
    auto upscale_factor_1d = context.mark_node(std::make_shared<v1::Reshape>(upscale_factor, neg_1, false));
    auto intermediate_shape = context.mark_node(
        std::make_shared<v0::Concat>(OutputVector{dims_before, neg_1, upscale_factor_1d, upscale_factor_1d, h, w}, 0));
    auto reshape = context.mark_node(std::make_shared<v1::Reshape>(x, intermediate_shape, false));
    // 2. Transpose tensor to [*, C, r, H, r, W]
    auto dims_before_len = context.mark_node(std::make_shared<v3::ShapeOf>(dims_before, element::i32));
    auto dims_before_len_s = context.mark_node(std::make_shared<v0::Squeeze>(dims_before_len, zero));
    auto order_begin = context.mark_node(std::make_shared<v4::Range>(zero_s, dims_before_len_s, one_s, element::i32));
    auto order_end_neg = context.mark_node(
        v0::Constant::create(element::i32, Shape{5}, {-3, 0, -2, 1, -1}));  // +2 because rank is expanded
    auto order_end = context.mark_node(std::make_shared<v1::Add>(order_end_neg, rank));
    auto order = context.mark_node(std::make_shared<v0::Concat>(OutputVector{order_begin, order_end}, 0));
    auto transpose = context.mark_node(std::make_shared<v1::Transpose>(reshape, order));
    // 3. Reshape to [*, -1, r * H, r * W]
    auto new_h = context.mark_node(std::make_shared<v1::Multiply>(h, upscale_factor));
    auto new_w = context.mark_node(std::make_shared<v1::Multiply>(w, upscale_factor));
    auto shape_after =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{dims_before, neg_1, new_h, new_w}, 0));
    return {context.mark_node(std::make_shared<v1::Reshape>(transpose, shape_after, false))};
};

OutputVector translate_pixel_unshuffle(const NodeContext& context) {
    // aten::pixel_unshuffle(Tensor self, int upscale_factor) -> Tensor
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto upscale_factor = get_input_as_i32(context, 1);
    auto neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto neg_3 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-3}));
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto zero_s = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto one_s = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    Output<Node> shape;
    Output<Node> rank;
    std::tie(shape, rank) = get_shape_rank(context, x, true);
    // 1. Reshape input to [-1, C, H / r, r, W / r, r], where r is upscale factor
    auto indices = context.mark_node(v0::Constant::create(element::i32, Shape{3}, {-3, -2, -1}));
    auto dims = context.mark_node(std::make_shared<v8::Gather>(shape, indices, zero_s));
    auto dims_splitted = context.mark_node(std::make_shared<v1::Split>(dims, zero_s, 3));
    auto c = dims_splitted->output(0);
    auto h = dims_splitted->output(1);
    auto w = dims_splitted->output(2);
    auto dims_before = context.mark_node(std::make_shared<v8::Slice>(shape, zero, neg_3, one));
    auto r = context.mark_node(std::make_shared<v0::Unsqueeze>(upscale_factor, zero));
    auto new_h = context.mark_node(std::make_shared<v1::Divide>(h, upscale_factor, true));
    auto new_w = context.mark_node(std::make_shared<v1::Divide>(w, upscale_factor, true));
    auto intermediate_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{neg_1, c, new_h, r, new_w, r}, 0));
    auto x_reshaped = context.mark_node(std::make_shared<v1::Reshape>(x, intermediate_shape, false));
    // 2. Transpose to [-1, C, r, r, H / r, W / r]
    auto transpose_order = context.mark_node(v0::Constant::create(element::i32, Shape{6}, {0, 1, 3, 5, 2, 4}));
    auto x_transposed = context.mark_node(std::make_shared<v1::Transpose>(x_reshaped, transpose_order));
    // 3. Reshape to [*, C*r*r, H / r, W / r]
    auto r_sqr = context.mark_node(std::make_shared<v1::Multiply>(r, r));
    auto new_c = context.mark_node(std::make_shared<v1::Multiply>(c, r_sqr));
    auto final_shape =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{dims_before, new_c, new_h, new_w}, 0));
    return {context.mark_node(std::make_shared<v1::Reshape>(x_transposed, final_shape, false))};
};

OutputVector translate_channel_shuffle(const NodeContext& context) {
    // aten::channel_shuffle(Tensor self, int groups) -> Tensor
    num_inputs_check(context, 2, 2);
    auto x = context.get_input(0);
    auto groups = context.get_input(1);
    auto neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto shape = context.mark_node(std::make_shared<v3::ShapeOf>(x, element::i32));
    // PyTorch realization uses assumption that channels dim is always 1
    auto indices = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {0, 1}));
    auto dims = context.mark_node(std::make_shared<v8::Gather>(shape, indices, zero));
    auto dims_splitted = context.mark_node(std::make_shared<v1::Split>(dims, zero, 2));
    auto c = dims_splitted->output(1);
    auto n = dims_splitted->output(0);
    groups = context.mark_node(std::make_shared<v0::Convert>(groups, element::i32));
    auto k = context.mark_node(std::make_shared<v1::Divide>(c, groups, true));
    auto g = context.mark_node(std::make_shared<v0::Unsqueeze>(groups, zero));
    // 1. Reshape input [N, G, K=C/G, -1]
    auto reshape_indices = context.mark_node(
        std::make_shared<v0::Concat>(OutputVector{std::move(n), std::move(g), std::move(k), std::move(neg_1)}, 0));
    x = context.mark_node(std::make_shared<v1::Reshape>(x, reshape_indices, false));
    // 2. Transpose to [N, K, G, -1]
    auto permute_indices = context.mark_node(v0::Constant::create(element::i32, Shape{4}, {0, 2, 1, 3}));
    auto y = context.mark_node(std::make_shared<v1::Transpose>(x, permute_indices));
    // 3. Reshape back to original shape
    auto result = context.mark_node(std::make_shared<v1::Reshape>(y, shape, false));
    return {result};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov