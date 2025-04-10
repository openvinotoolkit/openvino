// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/multiply.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_square_op(const NodeContext& node) {
    default_op_checks(node, 1, {"Square", "SQUARE"});

    auto n = node.get_input(0);
    auto res = make_shared<v1::Multiply>(n, n);
    set_node_name(node.get_name(), res);
    return res->outputs();
}

}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
