// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/less.hpp"
#include "openvino/op/non_zero.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
Output<Node> generate_indices_from_repeats_tensor(const NodeContext& context, const std::vector<int32_t>& repeats) {
    std::vector<int32_t> result;
    for (size_t i = 0; i < repeats.size(); ++i) {
        for (int32_t j = 0; j < repeats[i]; ++j) {
            result.push_back(static_cast<int32_t>(i));
        }
    }
    return context.mark_node(v0::Constant::create(element::i32, Shape{result.size()}, result));
};
}  // namespace

OutputVector translate_repeat_interleave(const NodeContext& context) {
    num_inputs_check(context, 2, 3);
    // constants
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto const_1_list = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));

    // inputs
    auto input = context.get_input(0);
    auto repeats = context.get_input(1);
    std::shared_ptr<ov::Node> result;

    auto repeats_ext = context.get_input_from_visible_context(1);
    auto repeats_const = ov::as_type_ptr<v0::Constant>(repeats_ext.get_node_shared_ptr());
    if (repeats_const && repeats_const->cast_vector<int32_t>().size() > 1) {
        // repeats is Constant with more then 1 element
        auto repeats = repeats_const->cast_vector<int32_t>();
        if (context.input_is_none(2)) {
            // case (repeats=tensor, dim=None)
            auto flat_shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
            auto reshape = context.mark_node(std::make_shared<v1::Reshape>(input, flat_shape, false));
            auto all_indices = generate_indices_from_repeats_tensor(context, repeats);
            result = std::make_shared<v8::Gather>(reshape, all_indices, const_0);
        } else {
            // case (repeats=tensor, dim=number)
            auto dimension = context.get_input(2);
            auto all_indices = generate_indices_from_repeats_tensor(context, repeats);
            result = std::make_shared<v8::Gather>(input, all_indices, dimension);
        }
    } else {
        repeats = context.mark_node(std::make_shared<v0::Convert>(repeats, element::i32));
        repeats = context.mark_node(std::make_shared<v1::Reshape>(repeats, const_neg_1, false));
        if (context.input_is_none(2)) {
            // case (repeats=number, dim=None)
            // Prepare the input and determine the maximum repeat value
            input = context.mark_node(std::make_shared<v1::Reshape>(input, const_neg_1, false));
            auto new_input = context.mark_node(std::make_shared<v0::Unsqueeze>(input, const_1));
            auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
            auto repeats_bc = context.mark_node(std::make_shared<v3::Broadcast>(repeats, input_shape));
            auto repeat_max = context.mark_node(std::make_shared<v1::ReduceMax>(repeats_bc, const_0, true));

            // Tile the input based on the maximum repeat value
            auto max_repeat_for_tile =
                context.mark_node(std::make_shared<v0::Concat>(OutputVector{const_1_list, repeat_max}, 0));
            auto tile = context.mark_node(std::make_shared<v0::Tile>(new_input, max_repeat_for_tile));
            auto tile_flat = context.mark_node(std::make_shared<v1::Reshape>(tile, const_neg_1, false));

            // Generate a range and compare to determine valid indices
            auto stop = context.mark_node(std::make_shared<v15::Squeeze>(repeat_max, const_0));
            auto range = context.mark_node(std::make_shared<v4::Range>(const_0, stop, const_1, element::i32));
            auto repeats_1 = context.mark_node(std::make_shared<v0::Unsqueeze>(repeats_bc, const_neg_1));
            auto less = context.mark_node(std::make_shared<v1::Less>(range, repeats_1));
            auto less_flat = context.mark_node(std::make_shared<v1::Reshape>(less, const_neg_1, false));

            // Identify non-zero indices from the comparison result
            auto non_zero = context.mark_node(std::make_shared<v3::NonZero>(less_flat));
            auto indices = context.mark_node(std::make_shared<v15::Squeeze>(non_zero, const_0));
            // Gather the final result using valid indices
            result = std::make_shared<v8::Gather>(tile_flat, indices, const_0);
        } else {
            // case (repeats=number, dim=number)
            // this only supports repeats as a single element tensor
            FRONT_END_CHECK_IMPLEMENTED(
                repeats.get_partial_shape().is_dynamic() || shape_size(repeats.get_partial_shape().to_shape()) == 1,
                "Repeats must be a single element tensor");
            repeats = context.mark_node(std::make_shared<v0::Concat>(OutputVector{repeats, const_1_list}, 0));
            auto shape_perm = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
            auto dimension = context.get_input(2);
            auto input_shape = context.mark_node(std::make_shared<v3::ShapeOf>(input, element::i32));
            auto input_dim_size = context.mark_node(std::make_shared<v8::Gather>(input_shape, dimension, const_0));
            auto range = context.mark_node(std::make_shared<v4::Range>(const_0, input_dim_size, const_1, element::i32));
            auto range_unsqeezed = context.mark_node(std::make_shared<v0::Unsqueeze>(range, const_0));
            auto tile = context.mark_node(std::make_shared<v0::Tile>(range_unsqeezed, repeats));
            auto transpose = context.mark_node(std::make_shared<v1::Transpose>(tile, shape_perm));
            auto flatten = context.mark_node(std::make_shared<v1::Reshape>(transpose, const_neg_1, false));
            result = std::make_shared<v8::Gather>(input, flatten, dimension);
        }
    }

    return {context.mark_node(result)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
