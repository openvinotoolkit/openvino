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

OutputVector translate_rsqrt_op(const NodeContext& node) {
    auto input = node.get_input(0);
    auto ng_exponent = make_shared<Constant>(input.get_element_type(), Shape{1}, -0.5f);
    auto res = make_shared<Power>(input, ng_exponent);
    set_node_name(node.get_name(), res);
    return res->outputs();
}
}  // namespace op
}  // namespace tensorflow
}  // namespace frontend
}  // namespace ov
