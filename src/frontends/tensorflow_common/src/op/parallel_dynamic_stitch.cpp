// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/reduce_max.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_parallel_dynamic_stitch_op(const NodeContext& node) {
    // format for inputs: [indices1, indices2, ..., indicesN, data1, data2, ..., dataN]
    // so we expect at least 2 input and the total number of inputs must be divisible by 2
    default_op_checks(node, 2, {"ParallelDynamicStitch", "DynamicStitch"});
    auto in_size = node.get_input_size();
    TENSORFLOW_OP_VALIDATION(node,
                             in_size % 2 == 0,
                             "The total number of inputs to DynamicStitch or ParallelDynamicStitch operation "
                             "must be divisible by 2.");

    int N = static_cast<int>(in_size / 2);
    OutputVector indices_to_concat;
    OutputVector data_to_concat;
    auto const_minus_one = make_shared<v0::Constant>(ov::element::i32, Shape{1}, -1);
    auto const_zero = make_shared<v0::Constant>(ov::element::i32, Shape{1}, 0);
    auto const_one = make_shared<v0::Constant>(ov::element::i32, Shape{1}, 1);
    for (int i = 0; i < N; ++i) {
        auto indices = node.get_input(i);
        auto data = node.get_input(N + i);

        const auto& indices_pshape = indices.get_partial_shape();
        auto rank = indices_pshape.rank();
        TENSORFLOW_OP_VALIDATION(node,
                                 indices_pshape.rank().is_static(),
                                 "Only static rank for `indices` input is supported.");
        auto rank_val = rank.get_length();
        auto norm_indices = make_shared<v1::Reshape>(indices, const_minus_one, false);
        if (rank_val < 1) {
            data = make_shared<v0::Unsqueeze>(data, const_zero);
        } else if (rank_val > 1) {
            auto data_shape = make_shared<v3::ShapeOf>(data, ov::element::i32);
            auto start = make_shared<v0::Constant>(ov::element::i32, Shape{1}, rank_val);
            auto stop = make_shared<v0::Constant>(ov::element::i32, Shape{1}, numeric_limits<int>::max());
            auto shape_of_single_element = make_shared<v8::Slice>(data_shape, start, stop, const_one);
            auto new_shape = make_shared<v0::Concat>(OutputVector{const_minus_one, shape_of_single_element}, 0);
            data = make_shared<v1::Reshape>(data, new_shape, false);
        }
        data_to_concat.push_back(data);
        indices_to_concat.push_back(norm_indices);
    }
    auto update = make_shared<v0::Concat>(data_to_concat, 0);
    auto indices = make_shared<v0::Concat>(indices_to_concat, 0);
    auto data_shape = make_shared<v3::ShapeOf>(update, ov::element::i32);

    auto zero = create_same_type_const_scalar<int32_t>(node.get_input(N), 0);
    auto zeros = make_shared<v3::Broadcast>(zero, data_shape);
    auto max_idx = make_shared<v1::ReduceMax>(indices, v0::Constant::create(element::i32, {1}, {0}), true);
    auto stop = make_shared<v1::Add>(max_idx->output(0), const_one);
    auto start = make_shared<v0::Constant>(ov::element::i32, Shape{1}, 0);
    auto axis = make_shared<v0::Constant>(ov::element::i32, Shape{1}, 0);
    auto sliced_zeros = make_shared<v8::Slice>(zeros, start, stop, const_one, axis);

    auto result = make_shared<v3::ScatterUpdate>(sliced_zeros, indices, update, const_zero);
    set_node_name(node.get_name(), result);
    return result->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
