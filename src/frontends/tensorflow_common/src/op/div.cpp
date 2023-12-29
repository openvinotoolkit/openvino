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
OutputVector translate_div_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Div"});
    auto x = node.get_input(0);
    auto y = node.get_input(1);
    bool m_pythondiv = false;

    // compute Division
    auto div = make_shared<v1::Divide>(x, y, m_pythondiv);

    set_node_name(node.get_name(), div);
    return div->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
