// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <climits>

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

OutputVector translate_transpose(NodeContext& context) {
    auto dim0 = context.const_input<int64_t>(1);
    auto dim1 = context.const_input<int64_t>(2);
    auto shape = std::make_shared<opset8::ShapeOf>(context.get_input(0), element::i32);
    auto rank_ = std::make_shared<opset8::ShapeOf>(shape, element::i32);
    auto rank = std::make_shared<opset8::Squeeze>(rank_);
    // Use opset::If for dim normalization
    auto dim0_node = context.get_input(1);
    auto dim1_node = context.get_input(2);
    if (dim0 < 0) {
        dim0_node = std::make_shared<opset8::Add>(rank, dim0_node);
    }
    if (dim1 < 0) {
        dim1_node = std::make_shared<opset8::Add>(rank, dim1_node);
    }
    auto start = opset8::Constant::create(element::i32, {}, {0});
    auto step = opset8::Constant::create(element::i32, {}, {1});
    auto range = std::make_shared<opset8::Range>(start, rank, step, element::i32);

    auto axis_0 = opset8::Constant::create(element::i64, Shape{}, {0});
    dim0_node = std::make_shared<opset8::Unsqueeze>(dim0_node, axis_0);
    dim1_node = std::make_shared<opset8::Unsqueeze>(dim1_node, axis_0);
    auto indices = std::make_shared<opset8::Concat>(OutputVector{dim0_node, dim1_node}, 0);
    auto updates = std::make_shared<opset8::Concat>(OutputVector{dim1_node, dim0_node}, 0);
    auto scatter = std::make_shared<opset8::ScatterElementsUpdate>(range, indices, updates, axis_0);

    /*auto data_pshape = context.get_input(0).get_partial_shape();
    auto rank = data_pshape.rank();
    OV_FRONTEND_REQUIRE(rank.is_static());
    auto _rank = rank.get_length();
    if (dim0 < 0) {
        dim0 = _rank + dim0;
    }
    if (dim1 < 0) {
        dim1 = _rank + dim1;
    }
    OV_FRONTEND_REQUIRE(dim0 > 0 && dim1 > 0);
    OV_FRONTEND_REQUIRE(dim0 < _rank && dim1 < _rank);
    std::vector<int64_t> order(_rank, 0);
    std::iota(order.begin(), order.end(), 0);
    std::swap(order[dim0], order[dim1]);
    auto order_const = context.mark_node(opset8::Constant::create(element::i64, {order.size()}, order));*/
    return {context.mark_node(std::make_shared<opset8::Transpose>(context.get_input(0), scatter))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov