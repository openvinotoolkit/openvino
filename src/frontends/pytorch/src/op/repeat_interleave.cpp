// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

namespace {
OutputVector generate_indices_from_repeats_tensor(std::vector<int64_t> repeats, NodeContext& context) {
    OutputVector all_indices;
    for (int i = 0; i < repeats.size(); i++) {
        Shape indices_shape{static_cast<size_t>(repeats.at(i))};
        std::vector<int64_t> indices_vec(repeats.at(i), i);
        auto indices = context.mark_node(opset10::Constant::create(element::i64, indices_shape, indices_vec));
        all_indices.push_back(indices);
    }
    return all_indices;
};
}  // namespace

OutputVector translate_repeat_interleave(NodeContext& context) {
    // constants
    auto const_0 = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {0}));
    auto const_1 = context.mark_node(opset10::Constant::create(element::i64, Shape{}, {1}));
    auto const_1_list = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {1}));
    auto const_neg_1 = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {-1}));

    // inputs
    auto input = context.get_input(0);
    std::shared_ptr<ov::Node> result;

    try {
        // repeats is Constant
        auto repeats = context.const_input<std::vector<int64_t>>(1);
        FRONT_END_OP_CONVERSION_CHECK(repeats.size() >= 1, "repeats should contain at least 1 element");
        auto const_repeats =
            context.mark_node(opset10::Constant::create(element::i64, Shape{2}, {repeats.at(0), (int64_t)1}));

        if (context.input_is_none(2)) {
            if (repeats.size() == 1) {
                // case (repeats=number, dim=None)
                auto flat_shape = context.mark_node(opset10::Constant::create(element::i64, Shape{2}, {1, -1}));
                auto reshape = context.mark_node(std::make_shared<opset10::Reshape>(input, flat_shape, false));
                auto tile = context.mark_node(std::make_shared<opset10::Tile>(reshape, const_repeats));
                auto shape_perm = context.mark_node(opset10::Constant::create(element::i64, Shape{2}, {1, 0}));
                auto transpose = context.mark_node(std::make_shared<opset10::Transpose>(tile, shape_perm));
                result = std::make_shared<opset10::Reshape>(transpose, const_neg_1, false);
            } else {
                // case (repeats=tensor, dim=None)
                auto flat_shape = context.mark_node(opset10::Constant::create(element::i64, Shape{1}, {-1}));
                auto reshape = context.mark_node(std::make_shared<opset10::Reshape>(input, flat_shape, false));
                OutputVector all_indices = generate_indices_from_repeats_tensor(repeats, context);
                auto concat = context.mark_node(std::make_shared<opset10::Concat>(all_indices, 0));
                result = std::make_shared<opset10::Gather>(reshape, concat, const_0);
            }
        } else {
            auto const_dim = context.get_input(2);
            if (repeats.size() == 1) {
                // case (repeats=number, dim=number)
                auto input_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(input, element::i64));
                auto input_dim_size =
                    context.mark_node(std::make_shared<opset10::Gather>(input_shape, const_dim, const_0));
                auto range =
                    context.mark_node(std::make_shared<opset10::Range>(const_0, input_dim_size, const_1, element::i64));
                auto range_unsqeezed = context.mark_node(std::make_shared<opset10::Unsqueeze>(range, const_0));
                auto tile = context.mark_node(std::make_shared<opset10::Tile>(range_unsqeezed, const_repeats));
                auto shape_perm =
                    context.mark_node(context.mark_node(opset10::Constant::create(element::i64, Shape{2}, {1, 0})));
                auto transpose = context.mark_node(std::make_shared<opset10::Transpose>(tile, shape_perm));
                auto flatten = context.mark_node(std::make_shared<opset10::Reshape>(transpose, const_neg_1, false));
                result = std::make_shared<opset10::Gather>(input, flatten, const_dim);
            } else {
                // case (repeats=tensor, dim=number)
                OutputVector all_indices = generate_indices_from_repeats_tensor(repeats, context);
                auto concat = context.mark_node(std::make_shared<opset10::Concat>(all_indices, 0));
                result = std::make_shared<opset10::Gather>(input, concat, const_dim);
            }
        }
    } catch (...) {
        // repeats is not Constant
        auto repeats_input = std::make_shared<opset10::Reshape>(context.get_input(1), const_1_list, false);
        auto repeats = std::make_shared<opset10::Concat>(OutputVector{repeats_input, const_1_list}, 0);
        if (context.input_is_none(2)) {
            // case (repeats=number, dim=None)
            auto flat_shape = context.mark_node(opset10::Constant::create(element::i64, Shape{2}, {1, -1}));
            auto reshape = context.mark_node(std::make_shared<opset10::Reshape>(input, flat_shape, false));
            auto tile = context.mark_node(std::make_shared<opset10::Tile>(reshape, repeats));
            auto shape_perm = context.mark_node(opset10::Constant::create(element::i64, Shape{2}, {1, 0}));
            auto transpose = context.mark_node(std::make_shared<opset10::Transpose>(tile, shape_perm));
            result = std::make_shared<opset10::Reshape>(transpose, const_neg_1, false);
        } else {
            // case (repeats=number, dim=number)
            auto const_dim = context.get_input(2);
            auto input_shape = context.mark_node(std::make_shared<opset10::ShapeOf>(input, element::i64));
            auto input_dim_size = context.mark_node(std::make_shared<opset10::Gather>(input_shape, const_dim, const_0));
            auto range =
                context.mark_node(std::make_shared<opset10::Range>(const_0, input_dim_size, const_1, element::i64));
            auto range_unsqeezed = context.mark_node(std::make_shared<opset10::Unsqueeze>(range, const_0));
            auto tile = context.mark_node(std::make_shared<opset10::Tile>(range_unsqeezed, repeats));
            auto shape_perm =
                context.mark_node(context.mark_node(opset10::Constant::create(element::i64, Shape{2}, {1, 0})));
            auto transpose = context.mark_node(std::make_shared<opset10::Transpose>(tile, shape_perm));
            auto flatten = context.mark_node(std::make_shared<opset10::Reshape>(transpose, const_neg_1, false));
            result = std::make_shared<opset10::Gather>(input, flatten, const_dim);
        }
    }

    return {context.mark_node(result)};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov
