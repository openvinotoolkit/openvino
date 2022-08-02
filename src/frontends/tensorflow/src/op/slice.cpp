// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_slice_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto start = node.get_input(1);
    auto size = node.get_input(2);

    auto size_const_in = get_constant_from_source(size);
    const auto& size_vector = size_const_in->cast_vector<int64_t>();

    auto start_const_in = get_constant_from_source(start);
    const auto& start_vector = start_const_in->cast_vector<int64_t>();

    std::vector<int64_t> stop_vector(size_vector.size());

    if (size_vector.size() != input.get_shape().size()) {
        FRONT_END_THROW("Slice size vector length is not equal to number of input dimensions.");
    }

    // Size value -1 => Slice from the "start" value to the end
    for (int i=0; i<size_vector.size(); i++) {
        if (size_vector[i] == -1) {
            stop_vector[i] = input.get_shape()[i];
        } else {
            stop_vector[i] = start_vector[i] + size_vector[i];
        }
    }

    auto start_const = make_shared<Constant>(element::i64, Shape{start_vector.size()}, start_vector);
    auto stop_const = make_shared<Constant>(element::i64, Shape{stop_vector.size()}, stop_vector);

    auto one = make_shared<Constant>(element::i64, Shape{1}, 1);
    auto shape = make_shared<ShapeOf>(start);
    auto step = make_shared<Broadcast>(one, shape);

    // If the input is zero-dim, convert to Const
    size_t input_dims = input.get_partial_shape().rank().get_length();
    if (input.get_partial_shape().is_static() && input_dims > 0 && input.get_shape()[0] == 0) {
        auto res = make_shared<Constant>(input.get_element_type(), ov::Shape{0}, std::vector<int>({0}));
        set_node_name(node.get_name(), res);
        return res->outputs();
    } else {
        auto res = make_shared<Slice>(input, start_const, stop_const, step);
        set_node_name(node.get_name(), res);
        return res->outputs();
    }
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
