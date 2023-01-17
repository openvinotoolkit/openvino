// Copyright (C) 2018-2023 Intel Corporation
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
OutputVector translate_broadcast_to_op(const NodeContext& node) {
    default_op_checks(node, 2, {"BroadcastTo"});
    auto input = node.get_input(0);
    auto shape = node.get_input(1);
    auto broadcast_to = make_shared<Broadcast>(input, shape);
    set_node_name(node.get_name(), broadcast_to);
    return {broadcast_to};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
