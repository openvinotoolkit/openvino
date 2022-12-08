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

OutputVector translate_cumsum_op(const NodeContext& node) {
    auto ng_x = node.get_input(0);
    auto ng_axis = node.get_input(1);
    auto exclusive = node.get_attribute<bool>("exclusive");
    auto reverse = node.get_attribute<bool>("reverse");

    auto res = make_shared<CumSum>(ng_x, ng_axis, exclusive, reverse);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov