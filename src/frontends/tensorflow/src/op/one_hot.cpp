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

OutputVector translate_one_hot_op(const NodeContext& node) {
    auto ng_features = node.get_input(0);
    auto ng_depth = node.get_input(1);
    auto ng_on = node.get_input(2);
    auto ng_off = node.get_input(3);

    auto one_hot_axis = node.get_attribute<int64_t>("axis");
    auto res = make_shared<OneHot>(ng_features, ng_depth, ng_on, ng_off, one_hot_axis);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
