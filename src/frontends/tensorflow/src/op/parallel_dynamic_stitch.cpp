// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset9.hpp"

using namespace std;
using namespace ov::opset9;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_parallel_dynamic_stitch_op(const NodeContext& node) {
    // format for inputs: [indices1, indices2, ..., indicesN, data1, data2, ..., dataN]
    // so we expect at least 2 input and the total number of inputs must be divisible by 2
    default_op_checks(node, 2, {"ParallelDynamicStitch", "DynamicStitch"});
    auto in_size = node.get_input_size();
    TENSORFLOW_OP_VALIDATION(node, in_size % 2 == 0, "The total number of inputs to DynamicStitch operation "
                                                     "must be divisible by 2.");

    size_t N = in_size/2;
    OutputVector indices_to_concat;
    OutputVector data_to_concat;
    auto data_element_type = node.get_input(N).get_element_type();
    auto const_minus_one = std::make_shared<Constant>(ov::element::i32, Shape{1}, -1);
    auto const_zero = std::make_shared<Constant>(ov::element::i32, Shape{1}, 0);
    auto const_one = std::make_shared<Constant>(ov::element::i32, Shape{1}, 1);
    for (size_t i = 0; i < N; ++i) {
        auto indices = node.get_input(static_cast<int>(i));
        auto data= node.get_input(static_cast<int>(N + i));

        const auto& indices_pshape = indices.get_partial_shape();
        auto rank = indices_pshape.rank();
        TENSORFLOW_OP_VALIDATION(node, indices_pshape.rank().is_static(),
                                 "Only static rank for `indices` input is supported.");
        auto rank_val = rank.get_length();
        auto norm_indices = std::make_shared<Reshape>(indices, const_minus_one, false);
        if (rank_val < 1) {
            data = std::make_shared<Unsqueeze>(data, const_zero);
        } else if (rank_val > 1) {
            auto data_shape = std::make_shared<ShapeOf>(data, ov::element::i32);
            auto start = std::make_shared<Constant>(ov::element::i32, Shape{1}, rank_val);
            auto stop = std::make_shared<Constant>(ov::element::i32, Shape{1}, std::numeric_limits<int>::max());
            auto shape_of_single_element = std::make_shared<Slice>(data_shape, start, stop, const_one);
            auto new_shape = std::make_shared<Concat>(OutputVector{const_minus_one, shape_of_single_element}, 0);
            data = std::make_shared<Reshape>(data, new_shape, false);
        }
        data_to_concat.push_back(data);
        indices_to_concat.push_back(norm_indices);
    }
    auto update = std::make_shared<Concat>(data_to_concat, 0);
    auto indices = std::make_shared<Concat>(indices_to_concat, 0);
    auto data_shape = std::make_shared<ShapeOf>(update, ov::element::i32);

    auto zero = std::make_shared<Constant>(data_element_type, Shape{1}, 0);
    auto zeros = std::make_shared<Broadcast>(zero, data_shape);

    auto scatter = std::make_shared<ScatterUpdate>(zeros, indices, update, const_zero);

    // we should cut zeros at the end of the result output to align it with tf framework
    auto max_idx = std::make_shared<TopK>(indices,
                                          Constant::create(element::i32, {}, {1}),
                                          0,
                                          TopK::Mode::MAX,
                                          TopK::SortType::SORT_VALUES);
    auto stop = std::make_shared<Add>(max_idx->output(0), const_one);
    auto start = std::make_shared<Constant>(ov::element::i32, Shape{1}, 0);
    auto axis = std::make_shared<Constant>(ov::element::i32, Shape{1}, 0);
    auto result = std::make_shared<Slice>(scatter, start, stop, const_one, axis);
    set_node_name(node.get_name(), result);
    return result->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
