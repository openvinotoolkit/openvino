// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/broadcast.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_fill_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Fill", "FILL"});
    auto dims = node.get_input(0);
    auto value = node.get_input(1);

    auto res = make_shared<v3::Broadcast>(value, dims);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
