// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

ov::OutputVector translate_size_op(const NodeContext& node) {
    // Size operation computes a number of elements in the input tensor
    default_op_checks(node, 1, {"Size"});
    auto input = node.get_input(0);

    // retrive attribute of the output type
    auto out_type = node.get_attribute<element::Type>("out_type", element::i32);

    // introduce extra dimension in order to compute size in case of a scalar input
    auto const_zero = make_shared<Constant>(element::i32, Shape{1}, 0);
    input = make_shared<Unsqueeze>(input, const_zero);

    // compute the input tensor size
    auto shape_of = make_shared<ShapeOf>(input, out_type);
    auto axis = make_shared<Constant>(element::i32, Shape{}, 0);
    auto size = make_shared<ReduceProd>(shape_of, axis);
    set_node_name(node.get_name(), size);
    return {size};
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
