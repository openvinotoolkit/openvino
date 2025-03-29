// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/pad.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
std::shared_ptr<Node> get_im2col_indices_along_dim(const NodeContext& context,
                                                   const Output<Node>& input_d,
                                                   int64_t kernel_size_d,
                                                   int64_t dilation_d,
                                                   int64_t padding_d,
                                                   int64_t stride_d) {
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{}, {-1}));
    auto kernel_size = context.mark_node(v0::Constant::create(element::i32, Shape{}, {kernel_size_d}));
    auto padding_2 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {padding_d * 2}));
    auto stride = context.mark_node(v0::Constant::create(element::i32, Shape{}, {stride_d}));
    auto input_d_squeezed = context.mark_node(std::make_shared<v0::Squeeze>(input_d, zero));
    auto blocks_d = context.mark_node(std::make_shared<v1::Add>(input_d_squeezed, padding_2));
    auto subtrahend =
        context.mark_node(v0::Constant::create(element::i32, Shape{}, {dilation_d * (kernel_size_d - 1)}));
    blocks_d = context.mark_node(std::make_shared<v1::Subtract>(blocks_d, subtrahend));
    auto blocks_d_indices = context.mark_node(std::make_shared<v4::Range>(zero, blocks_d, stride, element::i32));
    blocks_d_indices = context.mark_node(std::make_shared<v0::Unsqueeze>(blocks_d_indices, zero));
    std::vector<int64_t> rng;
    for (int64_t i = 0; i < kernel_size_d * dilation_d; i += dilation_d) {
        rng.push_back(i);
    }

    auto kernel_grid = context.mark_node(v0::Constant::create(element::i32, Shape{rng.size()}, rng));
    auto kernel_mask = context.mark_node(std::make_shared<v0::Unsqueeze>(kernel_grid, minus_one));
    return context.mark_node(std::make_shared<v1::Add>(blocks_d_indices, kernel_mask));
}
}  // namespace

OutputVector translate_im2col(const NodeContext& context) {
    num_inputs_check(context, 5, 5);
    auto input = context.get_input(0);
    auto kernel_size = context.const_input<std::vector<int64_t>>(1);
    PYTORCH_OP_CONVERSION_CHECK(kernel_size.size() == 2, "kernel size should contain 2 elements");
    auto dilation = context.const_input<std::vector<int64_t>>(2);
    PYTORCH_OP_CONVERSION_CHECK(dilation.size() == 2, "dilation should contain 2 elements");
    auto padding = context.const_input<std::vector<int64_t>>(3);
    PYTORCH_OP_CONVERSION_CHECK(padding.size() == 2, "padding should contain 2 elements");
    auto stride = context.const_input<std::vector<int64_t>>(4);
    PYTORCH_OP_CONVERSION_CHECK(stride.size() == 2, "stride should contain 2 elements");
    auto zero = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
    auto minus_one = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
    auto two = context.mark_node(v0::Constant::create(element::i32, Shape{}, {2}));
    auto four = context.mark_node(v0::Constant::create(element::i32, Shape{}, {4}));
    auto input_shape_split = context.mark_node(std::make_shared<v1::Split>(input_shape, zero, 4));
    const auto& input_b = input_shape_split->output(0);
    const auto& input_c = input_shape_split->output(1);
    const auto& input_h = input_shape_split->output(2);
    const auto& input_w = input_shape_split->output(3);
    const auto& stride_h = stride[0];
    const auto& stride_w = stride[1];
    const auto& padding_h = padding[0];
    const auto& padding_w = padding[1];
    const auto& dilation_h = dilation[0];
    const auto& dilation_w = dilation[1];
    const auto& kernel_h = kernel_size[0];
    const auto& kernel_w = kernel_size[1];
    auto blocks_row_indices = get_im2col_indices_along_dim(context, input_h, kernel_h, dilation_h, padding_h, stride_h);
    auto blocks_col_indices = get_im2col_indices_along_dim(context, input_w, kernel_w, dilation_w, padding_w, stride_w);
    auto kernel_window = context.mark_node(v0::Constant::create(element::i32, Shape{}, {kernel_h * kernel_w}));
    auto input_c_squeezed = context.mark_node(std::make_shared<v0::Squeeze>(input_c, zero));
    auto channel_unfolded = context.mark_node(std::make_shared<v1::Multiply>(input_c_squeezed, kernel_window));
    auto channel_unfolded_unsqueezed = context.mark_node(std::make_shared<v0::Unsqueeze>(channel_unfolded, zero));
    auto output_shape = context.mark_node(
        std::make_shared<v0::Concat>(OutputVector{input_b, channel_unfolded_unsqueezed, minus_one}, 0));
    auto pads = context.mark_node(
        v0::Constant::create(element::i32, Shape{4}, std::vector<int64_t>{0, 0, padding_h, padding_w}));
    auto zero_f = context.mark_node(std::make_shared<v1::ConvertLike>(zero, input));
    auto padded_input =
        context.mark_node(std::make_shared<v1::Pad>(input, pads, pads, zero_f, ov::op::PadMode::CONSTANT));
    auto output = context.mark_node(std::make_shared<v8::Gather>(padded_input, blocks_row_indices, two));
    output = context.mark_node(std::make_shared<v8::Gather>(output, blocks_col_indices, four));
    auto permutation_dims =
        context.mark_node(v0::Constant::create(element::i32, Shape{6}, std::vector<int64_t>{0, 1, 2, 4, 3, 5}));
    output = context.mark_node(std::make_shared<v1::Transpose>(output, permutation_dims));
    return {context.mark_node(std::make_shared<v1::Reshape>(output, output_shape, false))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov