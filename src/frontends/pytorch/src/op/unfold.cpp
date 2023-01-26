// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_unfold(NodeContext& context) {
    // constants
    auto const_0 = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {0}));
    auto const_1 = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {1}));
    auto const_0_list = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {0}));
    auto const_1_list = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {1}));
    auto const_neg_1_list = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {-1}));

    // get inputs and prepare auxiliary nodes
    auto input = context.get_input(0);
    auto sizes = context.mark_node(std::make_shared<opset10::ShapeOf>(input));
    auto input_rank = context.mark_node(std::make_shared<opset10::ShapeOf>(sizes));

    auto dimension =
        context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {context.const_input<int64_t>(1)}));
    auto dimension_plus_1 = context.mark_node(std::make_shared<opset10::Add>(dimension, const_1_list));

    int size_int = context.const_input<int64_t>(2);
    auto size_scalar = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {size_int}));
    auto size_list = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {size_int}));

    auto step = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {context.const_input<int64_t>(3)}));

    // calculate number of "slices"
    auto sizedim = context.mark_node(std::make_shared<opset10::Gather>(sizes, dimension, const_0));
    auto sizedim_minus_size = std::make_shared<opset10::Subtract>(sizedim, size_list);
    auto fraction = std::make_shared<opset10::Divide>(sizedim_minus_size, step);
    auto slices_count = std::make_shared<opset10::Add>(fraction, const_1);
    auto slices_count_scalar = context.mark_node(std::make_shared<opset10::Reshape>(slices_count, const_1, false));

    // generate indices for Gather
    auto start_indices_unscaled =
        context.mark_node(std::make_shared<opset10::Range>(const_0, slices_count_scalar, const_1, element::i64));
    auto start_indices = context.mark_node(std::make_shared<opset10::Multiply>(start_indices_unscaled, step));
    auto unsqueeze_indices = context.mark_node(std::make_shared<opset10::Unsqueeze>(start_indices, const_0));
    auto repeats_for_tile = context.mark_node(opset10::Constant::create(element::i64, Shape{2}, {size_int, 1}));
    auto tile = context.mark_node(std::make_shared<opset10::Tile>(unsqueeze_indices, repeats_for_tile));
    auto shape_perm = context.mark_node(context.mark_node(opset10::Constant::create(element::i64, Shape{2}, {1, 0})));
    auto transpose_indices = context.mark_node(std::make_shared<opset10::Transpose>(tile, shape_perm));
    auto range_zero_to_size =
        context.mark_node(std::make_shared<opset10::Range>(const_0, size_scalar, const_1, element::i64));
    auto correct_indices = context.mark_node(std::make_shared<opset10::Add>(transpose_indices, range_zero_to_size));
    auto reshape_indices =
        context.mark_node(std::make_shared<opset10::Reshape>(correct_indices, const_neg_1_list, false));

    // gather elements from input tensor
    auto gather = context.mark_node(std::make_shared<opset10::Gather>(input, reshape_indices, dimension));
    auto shape_begin =
        context.mark_node(std::make_shared<opset10::Slice>(sizes, const_0_list, dimension, const_1_list));
    auto shape_end =
        context.mark_node(std::make_shared<opset10::Slice>(sizes, dimension_plus_1, input_rank, const_1_list));
    auto required_shape = context.mark_node(
        std::make_shared<opset10::Concat>(OutputVector{shape_begin, slices_count, size_list, shape_end}, 0));
    auto reshape = context.mark_node(std::make_shared<opset10::Reshape>(gather, required_shape, false));

    // tranpose tensor with gathered element
    auto rank_scalar = context.mark_node(std::make_shared<opset10::Reshape>(input_rank, const_1, false));
    auto rank_plus_1_scalar = context.mark_node(std::make_shared<opset10::Add>(rank_scalar, const_1));
    auto dimension_plus_1_scalar =
        context.mark_node(std::make_shared<opset10::Reshape>(dimension_plus_1, const_1, false));
    auto dimension_plus_2_scalar = std::make_shared<opset10::Add>(dimension_plus_1_scalar, const_1);
    auto perm_begin =
        context.mark_node(std::make_shared<opset10::Range>(const_0, dimension_plus_1_scalar, const_1, element::i64));
    auto perm_end = context.mark_node(
        std::make_shared<opset10::Range>(dimension_plus_2_scalar, rank_plus_1_scalar, const_1, element::i64));
    auto perm =
        context.mark_node(std::make_shared<opset10::Concat>(OutputVector{perm_begin, perm_end, dimension_plus_1}, 0));
    auto transpose = context.mark_node(std::make_shared<opset10::Transpose>(reshape, perm));

    return {transpose};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
