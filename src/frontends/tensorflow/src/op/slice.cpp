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

    // compute stop values in case non-negative sizes
    auto stop_pos = make_shared<Add>(start, size);

    // compute stop values in case positive sizes
    auto input_shape = make_shared<ShapeOf>(input);
    auto one = make_shared<Constant>(size.get_element_type(), Shape{}, 1);
    auto stop_neg = make_shared<Add>(make_shared<Add>(make_shared<ConvertLike>(input_shape, size), size), one);

    // select the correct stop value based on a sign of size value
    auto zeros = make_shared<Constant>(size.get_element_type(), Shape{}, 0);
    auto negative_sizes_mask = make_shared<Less>(size, zeros);
    auto stop = make_shared<Select>(negative_sizes_mask, stop_neg, stop_pos);

    // broadcast step value
    auto start_shape = make_shared<ShapeOf>(start);
    auto step = make_shared<Broadcast>(one, start_shape);

    auto res = make_shared<Slice>(input, start, stop, step);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
