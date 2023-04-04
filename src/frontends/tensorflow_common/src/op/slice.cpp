// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"
#include "utils.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_slice_op(const NodeContext& node) {
    default_op_checks(node, 3, {"Slice"});
    auto input = node.get_input(0);
    auto start = node.get_input(1);
    auto size = node.get_input(2);

    // create axiliary constants
    auto const_one = create_same_type_const_scalar<int32_t>(start, 1);
    auto const_zero = create_same_type_const_scalar<int32_t>(start, 0);

    // compute stop values in case non-negative sizes
    auto stop_pos = make_shared<Add>(start, size);

    // compute stop values in case negative sizes
    // since TensorFlow supports only -1 among negative sizes
    // assign stop values to the data shape
    Output<Node> stop_neg = make_shared<ShapeOf>(input);
    stop_neg = make_shared<ConvertLike>(stop_neg, size);

    // select the correct stop value based on a sign of size value
    auto negative_sizes_mask = make_shared<Less>(size, const_zero);
    // TODO: investigate if we can simplify and replace Select with FloorMod operation
    // like FloorMod(size, input_shape)
    auto stop = make_shared<Select>(negative_sizes_mask, stop_neg, stop_pos);

    // broadcast step value
    auto start_shape = make_shared<ShapeOf>(start);
    auto step = make_shared<Broadcast>(const_one, start_shape);

    auto res = make_shared<Slice>(input, start, stop, step);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
