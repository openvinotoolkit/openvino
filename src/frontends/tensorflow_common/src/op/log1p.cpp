// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/opsets/opset8.hpp"

using namespace std;
using namespace ov::opset8;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_log_1p_op(const NodeContext& node) {
    // compute element-wise natural logarithm of (1 + x),
    // where x - input
    default_op_checks(node, 1, {"Log1p"});
    auto x = node.get_input(0);
    auto const_one = make_shared<Constant>(x.get_element_type(), Shape{}, 1);
    auto x_plus_one = make_shared<Add>(x, const_one);
    auto log1p = make_shared<Log>(x_plus_one);
    set_node_name(node.get_name(), log1p);
    return {log1p};
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
