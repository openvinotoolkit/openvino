// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_op_table.hpp"
#include "openvino/op/cum_sum.hpp"

using namespace std;
using namespace ov::op;

namespace ov {
namespace frontend {
namespace tensorflow {
namespace op {

OutputVector translate_cumsum_op(const NodeContext& node) {
    default_op_checks(node, 2, {"Cumsum"});
    auto x = node.get_input(0);
    auto axis = node.get_input(1);
    auto exclusive = node.get_attribute<bool>("exclusive", false);
    auto reverse = node.get_attribute<bool>("reverse", false);

    auto cum_sum = make_shared<v0::CumSum>(x, axis, exclusive, reverse);
    set_node_name(node.get_name(), cum_sum);
    return cum_sum->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
