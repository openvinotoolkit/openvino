// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/op/transpose.hpp"

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/range.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_transpose(NodeContext& context) {
    num_inputs_check(context, 3, 3);
    auto dim0 = context.const_input<int64_t>(1);
    auto dim1 = context.const_input<int64_t>(2);
    Output<Node> rank;
    std::tie(std::ignore, rank) = get_shape_rank(context, context.get_input(0), true);
    // Use opset::If for dim normalization
    auto dim0_node = context.get_input(1);
    auto dim1_node = context.get_input(2);
    if (dim0 < 0) {
        dim0_node = std::make_shared<v1::Add>(rank, dim0_node);
    }
    if (dim1 < 0) {
        dim1_node = std::make_shared<v1::Add>(rank, dim1_node);
    }
    auto start = v0::Constant::create(element::i32, {}, {0});
    auto step = v0::Constant::create(element::i32, {}, {1});
    auto range = std::make_shared<v4::Range>(start, rank, step, element::i32);

    auto axis_0 = v0::Constant::create(element::i64, Shape{}, {0});
    auto dim0_node_ = std::make_shared<v0::Unsqueeze>(dim0_node, axis_0);
    auto dim1_node_ = std::make_shared<v0::Unsqueeze>(dim1_node, axis_0);
    auto indices = std::make_shared<v0::Concat>(OutputVector{dim0_node_, dim1_node_}, 0);
    auto updates = std::make_shared<v0::Concat>(OutputVector{dim1_node_, dim0_node_}, 0);
    auto scatter = std::make_shared<v3::ScatterElementsUpdate>(range, indices, updates, axis_0);
    context.mark_nodes({start, step, range, axis_0, dim0_node_, dim1_node_, indices, updates, scatter});

    return {context.mark_node(std::make_shared<v1::Transpose>(context.get_input(0), scatter))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov