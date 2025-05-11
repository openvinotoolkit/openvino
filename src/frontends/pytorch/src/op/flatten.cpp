// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "openvino/frontend/pytorch/node_context.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

namespace ov {
namespace frontend {
namespace pytorch {
namespace op {

using namespace ov::op;

OutputVector translate_flatten(const NodeContext& context) {
    num_inputs_check(context, 1, 3);
    auto x = context.get_input(0);
    Output<Node> shape;
    Output<Node> rank;
    std::tie(shape, rank) = get_shape_rank(context, x, true);
    // Use opset::If for dim normalization. For now we only have flatten with constant start and end
    Output<Node> start_dim_node;
    Output<Node> end_dim_node;
    if (!context.input_is_none(1)) {
        start_dim_node = get_input_as_i32(context, 1);
    } else {
        start_dim_node = v0::Constant::create(element::i32, Shape{}, {0});
    }
    if (!context.input_is_none(2)) {
        end_dim_node = get_input_as_i32(context, 2);
    } else {
        end_dim_node = v0::Constant::create(element::i32, Shape{}, {-1});
    }
    start_dim_node = normalize_axis(context, start_dim_node, rank);
    end_dim_node = normalize_axis(context, end_dim_node, rank);
    // Slice shape from begin and end, then concat with -1, if slice return empty tensor concat should still be able to
    // work with it
    auto zero = v0::Constant::create(element::i32, Shape{1}, {0});
    auto one = v0::Constant::create(element::i32, Shape{1}, {1});
    auto int_max = v0::Constant::create(element::i32, Shape{1}, {std::numeric_limits<int32_t>::max()});
    auto start_dim_u = std::make_shared<v0::Unsqueeze>(start_dim_node, zero);
    auto slice_begin = std::make_shared<v8::Slice>(shape, zero, start_dim_u, one);
    auto neg_1_const = v0::Constant::create(element::i32, Shape{1}, {-1});
    auto end_dim_u = std::make_shared<v0::Unsqueeze>(end_dim_node, zero);
    auto end_dim_next = std::make_shared<v1::Add>(end_dim_u, one);
    auto slice_end = std::make_shared<v8::Slice>(shape, end_dim_next, int_max, one);
    auto new_shape = std::make_shared<v0::Concat>(OutputVector{slice_begin, neg_1_const, slice_end}, 0);

    context.mark_nodes({zero, one, int_max, start_dim_u, end_dim_u, slice_begin, slice_end, neg_1_const, new_shape});

    return {context.mark_node(std::make_shared<v1::Reshape>(x, new_shape, true))};
};

}  // namespace op
}  // namespace pytorch
}  // namespace frontend
}  // namespace ov