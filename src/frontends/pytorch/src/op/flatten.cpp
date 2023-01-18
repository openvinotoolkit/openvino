// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset10.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_flatten(NodeContext& context) {
    auto start_dim = context.const_input<int64_t>(1);
    auto end_dim = context.const_input<int64_t>(2);

    auto shape = std::make_shared<opset10::ShapeOf>(context.get_input(0), element::i32);
    auto rank_ = std::make_shared<opset10::ShapeOf>(shape, element::i32);
    auto rank = std::make_shared<opset10::Squeeze>(rank_);
    // Use opset::If for dim normalization
    auto start_dim_node = context.get_input(1);
    auto end_dim_node = context.get_input(2);
    if (start_dim < 0) {
        start_dim_node = std::make_shared<opset10::Add>(rank, start_dim_node);
    }
    if (end_dim < 0) {
        end_dim_node = std::make_shared<opset10::Add>(rank, end_dim_node);
    }
    auto delta = std::make_shared<opset10::Subtract>(end_dim_node, start_dim_node);
    auto rank_delta = std::make_shared<opset10::Subtract>(rank, delta);
    auto true_const0 = opset10::Constant::create(element::boolean, Shape{}, {1});
    auto zeros_loop = std::make_shared<opset10::Loop>(rank_delta, true_const0);
    auto true_const = opset10::Constant::create(element::boolean, Shape{}, {1});
    auto result_true = std::make_shared<opset10::Result>(true_const);
    auto zero_const = opset10::Constant::create(element::i32, Shape{1}, {0});
    auto result_zero = std::make_shared<opset10::Result>(zero_const);
    auto f = std::make_shared<ov::Model>(ResultVector{result_true, result_zero}, ParameterVector{});
    zeros_loop->set_function(f);
    zeros_loop->set_special_body_ports({-1, 0});
    auto zeros = zeros_loop->get_concatenated_slices(result_zero, 0, 1, 1, -1, 0);
    auto neg_1_const = opset10::Constant::create(element::i32, Shape{1}, {-1});
    auto axis_0 = opset10::Constant::create(element::i32, Shape{1}, {0});
    auto start_dim_node_ = std::make_shared<opset10::Unsqueeze>(start_dim_node, axis_0);
    auto new_shape = std::make_shared<opset10::ScatterElementsUpdate>(zeros, start_dim_node_, neg_1_const, axis_0);

    context.mark_nodes({shape,
                        rank_,
                        rank,
                        delta,
                        rank_delta,
                        true_const0,
                        zeros_loop,
                        neg_1_const,
                        axis_0,
                        start_dim_node_,
                        new_shape});

    return {context.mark_node(std::make_shared<opset10::Reshape>(context.get_input(0), new_shape, true))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov