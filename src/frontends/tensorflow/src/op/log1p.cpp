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

OutputVector translate_log_1p_op(const NodeContext& node) {
    auto n = node.get_input(0);
    auto const_1 = make_shared<Constant>(n.get_element_type(), Shape{}, 1);
    auto add = make_shared<Add>(n, const_1);
    auto res = make_shared<Log>(add);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
