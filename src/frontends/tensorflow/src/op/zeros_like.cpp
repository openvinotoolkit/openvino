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

OutputVector translate_zeros_like_op(const NodeContext& node) {
    auto x = node.get_input(0);
    auto shape_of = make_shared<ShapeOf>(x);
    auto zero = make_shared<Constant>(x.get_element_type(), Shape{1}, 0);
    auto res = make_shared<Broadcast>(zero, shape_of);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
