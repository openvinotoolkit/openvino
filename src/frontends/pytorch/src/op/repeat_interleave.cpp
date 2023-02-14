// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "pt_framework_node.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

namespace {
OutputVector generate_indices_from_repeats_tensor(const NodeContext& context, const std::vector<int32_t>& repeats) {
    OutputVector all_indices;
    for (size_t i = 0; i < repeats.size(); i++) {
        Shape indices_shape{static_cast<size_t>(repeats.at(i))};
        std::vector<int32_t> indices_vec(repeats.at(i), static_cast<int32_t>(i));
        auto indices = context.mark_node(v0::Constant::create(element::i32, indices_shape, indices_vec));
        all_indices.push_back(indices);
    }
    return all_indices;
};
}  // namespace

OutputVector translate_repeat_interleave(NodeContext& context) {
    num_inputs_check(context, 2, 3);
    // constants
    auto const_0 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {0}));
    auto const_1 = context.mark_node(v0::Constant::create(element::i32, Shape{}, {1}));
    auto const_1_list = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {1}));
    auto const_neg_1 = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));

    // inputs
    auto input = context.get_input(0);
    std::shared_ptr<ov::Node> result;

    auto repeats_ext_node = context.get_input_from_visible_context(1).get_node_shared_ptr();
    auto repeats_fw_node = std::dynamic_pointer_cast<v0::Constant>(repeats_ext_node);
    if (repeats_fw_node && repeats_fw_node->cast_vector<int32_t>().size() > 1) {
        // repeats is Constant with more then 1 element
        auto repeats = repeats_fw_node->cast_vector<int32_t>();
        if (context.input_is_none(2)) {
            // case (repeats=tensor, dim=None)
            auto flat_shape = context.mark_node(v0::Constant::create(element::i32, Shape{1}, {-1}));
            auto reshape = context.mark_node(std::make_shared<v1::Reshape>(input, flat_shape, false));
            OutputVector all_indices = generate_indices_from_repeats_tensor(context, repeats);
            auto concat = context.mark_node(std::make_shared<v0::Concat>(all_indices, 0));
            result = std::make_shared<v8::Gather>(reshape, concat, const_0);
        } else {
            // case (repeats=tensor, dim=number)
            auto dimension = context.get_input(2);
            OutputVector all_indices = generate_indices_from_repeats_tensor(context, repeats);
            auto concat = context.mark_node(std::make_shared<v0::Concat>(all_indices, 0));
            result = std::make_shared<v8::Gather>(input, concat, dimension);
        }
    } else {
        // repeats is not Constant or single element constant
        // Curently we support only case when repeats contains only one element. Otherwise next Reshape will fail.
        auto repeats_input =
            context.mark_node(std::make_shared<v1::Reshape>(context.get_input(1), const_1_list, false));
        auto repeats = context.mark_node(std::make_shared<v0::Concat>(OutputVector{repeats_input, const_1_list}, 0));
        auto shape_perm = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, 0}));
        if (context.input_is_none(2)) {
            // case (repeats=number, dim=None)
            auto flat_shape = context.mark_node(v0::Constant::create(element::i32, Shape{2}, {1, -1}));
            auto reshape = context.mark_node(std::make_shared<v1::Reshape>(input, flat_shape, false));
            auto tile = context.mark_node(std::make_shared<v0::Tile>(reshape, repeats));
            auto transpose = context.mark_node(std::make_shared<v1::Transpose>(tile, shape_perm));
            result = std::make_shared<v1::Reshape>(transpose, const_neg_1, false);
        } else {
            // case (repeats=number, dim=number)
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
