// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "op_table.hpp"
#include "openvino/opsets/opset10.hpp"

using namespace std;
using namespace ov::opset10;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_is_inf_op(const NodeContext& node) {
    default_op_checks(node, 1, {"IsInf"});
    auto x = node.get_input(0);

    auto is_inf = make_shared<IsInf>(x);
    set_node_name(node.get_name(), is_inf);
    return {is_inf};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
