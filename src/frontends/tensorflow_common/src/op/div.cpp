// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/divide.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {
OutputVector translate_binary_op(const NodeContext& node,
                                 const std::function<shared_ptr<v1::Divide>(Output<Node>&, Output<Node>&)>& Div) {
    default_op_checks(node, 2, {"Div"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);
    auto result = Div(x, y);
    set_node_name(node.get_name(), result);
    return {result};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
