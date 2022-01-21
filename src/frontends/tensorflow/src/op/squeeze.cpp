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

OutputVector translate_squeeze_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto axes = node.get_attribute<std::vector<int64_t>>("squeeze_dims");
    auto axes_const = make_shared<Constant>(element::i32, Shape{axes.size()}, axes);
    auto res = make_shared<Squeeze>(input, axes_const);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov