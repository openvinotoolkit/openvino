// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace ov::op;

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_unfold(const NodeContext& context) {
    num_inputs_check(context, 4, 4);
    // constants
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto const_0_list = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {0}));
    auto const_1_list = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_neg_1_list = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));

    // get inputs and prepare auxiliary nodes
    auto input = context.get_input(0);
    Output<Node> input_shape;
    Output<Node> input_rank;
    std::tie(input_shape, input_rank) = get_shape_rank(context, input);

    auto dimension = context.mark_node(std::make_shared<v0::Unsqueeze>(context.get_input(1), const_0));
    dimension = context.mark_node(std::make_shared<v0::Convert>(dimension, element::i32));
    auto dimension_plus_1 = context.mark_node(std::make_shared<v1::Add>(dimension, const_1_list));

    auto size_scalar = get_input_as_i32(context, 2);
    auto size_list = context.mark_node(std::make_shared<v1::Reshape>(size_scalar, const_1_list, false));

    auto step = get_input_as_i32(context, 3);

    // calculate number of "slices"
    auto sizedim = context.mark_node(std::make_shared<v8::Gather>(input_shape, dimension, const_0));
    auto sizedim_minus_size = std::make_shared<v1::Subtract>(sizedim, size_list);
    auto fraction = std::make_shared<v1::Divide>(sizedim_minus_size, step);
    auto slices_count = std::make_shared<v1::Add>(fraction, const_1);
    auto slices_count_scalar = context.mark_node(std::make_shared<v1::Reshape>(slices_count, const_1, false));

    // generate indices for Gather, i.e.:
    // [0,1,...,size-1, 0+step, 1+step,..., size-1+step,..., (slices_count-1)*step,..., size-1+(slices_count-1)*step]
    // for example [0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 6, 7] (size=4, step=2, slices_count=3)
    auto start_indices_unscaled =
        context.mark_node(std::make_shared<v4::Range>(const_0, slices_count_scalar, const_1, element::i32));
    auto start_indices = context.mark_node(std::make_shared<v1::Multiply>(start_indices_unscaled, step));
    auto unsqueeze_indices = context.mark_node(std::make_shared<v0::Unsqueeze>(start_indices, const_0));
    auto repeats_for_tile = context.mark_node(std::make_shared<v0::Concat>(OutputVector{size_list, const_1_list}, 0));
    auto tile = context.mark_node(std::make_shared<v0::Tile>(unsqueeze_indices, repeats_for_tile));
    auto shape_perm = context.mark_node(context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0})));
    auto transpose_indices = context.mark_node(std::make_shared<v1::Transpose>(tile, shape_perm));
    auto range_zero_to_size =
        context.mark_node(std::make_shared<v4::Range>(const_0, size_scalar, const_1, element::i32));
    auto correct_indices = context.mark_node(std::make_shared<v1::Add>(transpose_indices, range_zero_to_size));
    auto reshape_indices = context.mark_node(std::make_shared<v1::Reshape>(correct_indices, const_neg_1_list, false));

    // gather elements from input tensor
    auto gather = context.mark_node(std::make_shared<v8::Gather>(input, reshape_indices, dimension));
    auto shape_begin =
        context.mark_node(std::make_shared<v8::Slice>(input_shape, const_0_list, dimension, const_1_list));
    auto shape_end =
        context.mark_node(std::make_shared<v8::Slice>(input_shape, dimension_plus_1, input_rank, const_1_list));
    auto required_shape = context.mark_node(
        std::make_shared<v0::Concat>(OutputVector{shape_begin, slices_count, size_list, shape_end}, 0));
    auto reshape = context.mark_node(std::make_shared<v1::Reshape>(gather, required_shape, false));

    // transpose tensor with gathered element
    auto rank_scalar = context.mark_node(std::make_shared<v1::Reshape>(input_rank, const_1, false));
    auto rank_plus_1_scalar = context.mark_node(std::make_shared<v1::Add>(rank_scalar, const_1));
    auto dimension_plus_1_scalar = context.mark_node(std::make_shared<v1::Reshape>(dimension_plus_1, const_1, false));
    auto dimension_plus_2_scalar = std::make_shared<v1::Add>(dimension_plus_1_scalar, const_1);
    auto perm_begin =
        context.mark_node(std::make_shared<v4::Range>(const_0, dimension_plus_1_scalar, const_1, element::i32));
    auto perm_end = context.mark_node(
        std::make_shared<v4::Range>(dimension_plus_2_scalar, rank_plus_1_scalar, const_1, element::i32));
    auto perm =
        context.mark_node(std::make_shared<v0::Concat>(OutputVector{perm_begin, perm_end, dimension_plus_1}, 0));
    auto transpose = context.mark_node(std::make_shared<v1::Transpose>(reshape, perm));

    return {transpose};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
