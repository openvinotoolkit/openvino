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

    auto stop = make_shared<Add>(start, size);

    auto one = make_shared<Constant>(element::i64, Shape{1}, 1);
    auto shape = make_shared<ShapeOf>(start);
    auto step = make_shared<Broadcast>(one, shape);

    auto res = make_shared<Slice>(input, start, stop, step);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
