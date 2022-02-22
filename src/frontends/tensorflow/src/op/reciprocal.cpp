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

OutputVector translate_reciprocal_op(const NodeContext& node) {
    auto x = node.get_input(0);
    auto ng_exponent = make_shared<Constant>(x.get_element_type(), Shape{}, -1);
    auto res = make_shared<Power>(x, ng_exponent);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
